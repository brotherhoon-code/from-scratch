import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import itertools

# yolo v1의 bbone으로 사용되는 architecture입니다.


def conv2d_bn(in_c, out_c, k_size, s=1, p=0, bn=True):
    list_layers = []
    list_layers.append(nn.Conv2d(in_c, out_c, k_size, s, p))
    if bn:
        list_layers.append(nn.BatchNorm2d(out_c))
    list_layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*list_layers)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, x:torch.Tensor):
        return x.squeeze()




class DarkNet(nn.Module):
    def __init__(self, is_cls_head=False, bn=True):
        super(DarkNet, self).__init__()
        self.is_cls_head = is_cls_head
        self.bbone = self._make_layers(bn=bn)
        self.cls_head = self._make_classifier() if is_cls_head else nn.Sequential()
        
    def forward(self, x):
        x = self.bbone(x)
        if self.is_cls_head:
            x = self.cls_head(x)
        return x
    
    def _make_layers(self, bn:bool=True):
        list_layers = [
            conv2d_bn(3, 64,7,s=2,p=3, bn=bn),
            nn.MaxPool2d(2), 
            
            conv2d_bn(64, 192, 3, p=1, bn=bn),
            nn.MaxPool2d(2), 

            conv2d_bn(192, 128, 1, bn=bn),
            conv2d_bn(128, 256, 3, p=1, bn=bn),
            conv2d_bn(256, 256, 1, bn=bn),
            conv2d_bn(256, 512, 3, p=1, bn=bn),
            nn.MaxPool2d(2),
            
            conv2d_bn(512, 256, 1, bn=bn),
            conv2d_bn(256, 512, 3, p=1, bn=bn),
            conv2d_bn(512, 256, 1, bn=bn),
            conv2d_bn(256, 512, 3, p=1, bn=bn),
            conv2d_bn(512, 256, 1, bn=bn),
            conv2d_bn(256, 512, 3, p=1, bn=bn),
            conv2d_bn(512, 256, 1, bn=bn),
            conv2d_bn(256, 512, 3, p=1, bn=bn),
            conv2d_bn(512, 512, 1, bn=bn),
            conv2d_bn(512, 1024, 3, p=1, bn=bn),
            nn.MaxPool2d(2),
             
            conv2d_bn(1024, 512, 1, bn=bn),
            conv2d_bn(512, 1024, 3, p=1, bn=bn),
            conv2d_bn(1024, 512, 1, bn=bn),
            conv2d_bn(512, 1024, 3, p=1,bn=bn),
            ]
        return nn.Sequential(*list_layers)
    
    def _make_classifier(self):
        list_layers = []
        list_layers.append(nn.AvgPool2d(7))
        list_layers.append(Squeeze())
        list_layers.append(nn.Linear(1024, 1000))
        return nn.Sequential(*list_layers)



if __name__  == "__main__":
    darknet = DarkNet(is_cls_head=True, bn=True)
    summary(darknet, (3,224,224), batch_size=2, device='cpu')