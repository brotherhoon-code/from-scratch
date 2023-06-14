import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

# BaseModel의 추상화 메서드 train_step, val_step, test_step를 구현해야 함
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
    
    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss': # 사전 정의
            return {'loss':F.cross_entropy(x, labels)} # 사전 정의
        elif mode == 'predict':
            return x, labels
    
    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
    
    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs