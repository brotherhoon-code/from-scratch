'''
reference: https://github.com/Jasonlee1995/DeepLab_v1
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as F

import itertools
import numpy as np

import pydensecrf.densecrf as dcrf
import pydensecrf.utils

from torchsummary import summary
from PIL import Image


class DenseCRF():
    def __init__(self, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.iter_max = iter_max
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std

    def __call__(self, image, prob_map):
        C, H, W = prob_map.shape
        
        image = image.permute((1, 2, 0))
        prob_map = prob_map.cpu().numpy()
        
        U = pydensecrf.utils.unary_from_softmax(prob_map)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w)

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def make_layers()->list:
    """
    vgg16의 architecture를 일부 수정하여 리턴합니다.
    - 수정범위: conv5의 패딩 스트라이드

    Returns:
        list: [conv1, conv2, conv3, conv4, conv5]
    """
    vgg16 = models.vgg16(pretrained=False)
    backbone = list(vgg16.features.children())
    conv1 = nn.Sequential(*backbone[:4]) # conv:0-1-2-3: maxpool:4(del)
    conv2 = nn.Sequential(*backbone[5:9])
    conv3 = nn.Sequential(*backbone[10:16])
    conv4 = nn.Sequential(*backbone[17:23])
    conv5 = nn.Sequential(*backbone[24:30])
    for i in range(len(conv5)): # 5번째 convlayer의 padding과 dilation을 변경
        if isinstance(conv5[i], nn.Conv2d):
            conv5[i].padding=(2,2)
            conv5[i].dilation=(2,2)   
    return [conv1, conv2, conv3, conv4, conv5]


class VGG16_LargeFOV(nn.Module):
    """
    DeepLab V1에 맞춰서 만들어진 backbone입니다.
    pool3 이후로부터는 spatial에 변화가 없습니다.
    """
    def __init__(self,
                 num_classes:int,
                 init_weights:bool=False):
        super().__init__()
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = make_layers()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv67 = nn.Sequential(nn.Conv2d(in_channels=512,
                                              out_channels=1024,
                                              kernel_size=3,
                                              stride=1,
                                              padding=12,
                                              dilation=12),
                                    nn.ReLU(inplace=True), 
                                    nn.Dropout2d(), 
                                    
                                    nn.Conv2d(in_channels=1024,
                                              out_channels=1024,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0), 
                                    nn.ReLU(inplace=True), 
                                    nn.Dropout2d(), 
                                    
                                    nn.Conv2d(in_channels=1024,
                                              out_channels=num_classes,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        if init_weights:
            self._initialize_weight()
        
    def _initialize_weight(self):
        targets = [self.conv67]
        for layer in targets:
            for module in layer:
                if isinstance(module, nn.Conv2d):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x): 
        # input 3, 224, 224
        
        x = self.conv1(x)
        x = self.pool1(x)
        # 64, 112, 112
        
        x = self.conv2(x)
        x = self.pool2(x)
        # 128, 56, 56
        
        x = self.conv3(x)
        x = self.pool3(x)
        # 256, 28, 28
        
        x = self.conv4(x)
        x = self.pool4(x)
        # 512, 28, 28
        
        x = self.conv5(x)
        x = self.pool5(x)
        # 512, 28, 28
        
        x = self.pool(x)
        # 512, 28, 28
        
        x = self.conv67(x)
        # n_classes, 28, 28
        
        return x


class DeepLab_v1():
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = VGG16_LargeFOV(self.num_classes)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
    def forward(self, x):
        """
        forward 과정은 binear upsampling 만 존재합니다.
        """
        B, C, H, W = x.shape
        x = self.model(x)
        x = F.resize(x, (H, W), Image.BILINEAR)
        return x
    
    def inference(self,
                  image_dir,
                  iter_max,
                  bi_w,
                  bi_xy_std,
                  bi_rgb_std,
                  pos_w,
                  pos_xy_std):
        """
        1. 이미지를 읽어와서 DL모델을 이용해 작은 resolution으로 pred
        2. BILINEAR을 사용 upsampling
        3. DenseDRF(후처리)
        """
        self.model.eval()
        with torch.no_grad():
            image = Image.open(image_dir).convert('RGB') # H W C
            image_tensor = torch.as_tensor(np.asarray(image)) # H W C
            image_tensor = image_tensor.view(image.size[1], image.size[0], len(image.getbands())) # W H C?
            image_tensor = image_tensor.permute((2,0,1)) # C W H?
            
            C, H, W = image_tensor.shape
            image_norm_tensor = image_tensor[None, ...].float().div(255)
            image_norm_tensor = image_norm_tensor.sub_(self.mean).div_(self.std)
            
            output = self.model(image_norm_tensor)
            output = F.resize(output, (H,W), Image.BILINEAR)
            output = nn.Softmax2d()(output)
            output = output[0]
            
            crf = DenseCRF(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
            
            predict = crf(image_tensor, output)
            predict = np.argmax(predict, axis=0)
            
            return predict
        
        

if __name__ == "__main__":
    m = DeepLab_v1(10)
    m.forward(torch.Tensor(1,3,224,224))