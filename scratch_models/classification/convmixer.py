import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


'''
ref: https://github.com/locuslab/convmixer/blob/main/convmixer.py

## 1. 모델 설계  
ConvMixer
    PatchEmbedBlock
    MixerBlock
        SpatialMixBlock
        ChannelMixBlock

## 2. 구현 포인트  
isotropic 형태로 간단한 모델입니다.

## 3. 주의 사항  
skip conn은 MixerBlock에 구현되어 있습니다.
'''

class PatchEmbedBlock(nn.Module):
    def __init__(self, 
                 dim:int, 
                 patch_size:int):
        super(PatchEmbedBlock, self).__init__()
        self.dim=dim
        self.patch_size=patch_size
        self.embed_layer = nn.Conv2d(in_channels=3, 
                                     out_channels=dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
        
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x # b dim h/patch_size w/patch_size

    
class SpatialMixBlock(nn.Module):
    def __init__(self, 
                 dim:int, 
                 kernel_size:int):
        super(SpatialMixBlock, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.spatial_mix_layer = nn.Conv2d(in_channels=dim, 
                                           out_channels=dim, 
                                           kernel_size=kernel_size, 
                                           groups=dim, 
                                           padding="same")
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
        
    def forward(self, x):
        x = self.spatial_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x

    
class ChannelMixBlock(nn.Module):
    def __init__(self, 
                 dim:int):
        super(ChannelMixBlock, self).__init__()
        self.dim = dim
        self.channel_mix_layer = nn.Conv2d(dim, dim ,kernel_size=(1,1))
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
    
    def forward(self, x):
        x = self.channel_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x
    
class MixerBlock(nn.Module):
    def __init__(self, 
                 dim:int, 
                 kernel_size:int):
        super(MixerBlock, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.spatial_mix_block = SpatialMixBlock(dim, kernel_size)
        self.channel_mix_block = ChannelMixBlock(dim)
    
    def forward(self, x):
        identity = x
        x = self.spatial_mix_block(x) + identity
        x = self.channel_mix_block(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, 
                 dim:int, 
                 depth: int, 
                 patch_size:int, 
                 kernel_size:int, 
                 n_classes:int):
        super(ConvMixer, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.depth = depth
        self.kernel_Size = kernel_size
        
        self.patch_embed_layer = PatchEmbedBlock(dim, patch_size)
        self.mixer_layers = nn.Sequential(*[MixerBlock(dim, kernel_size)]*depth)
        self.neck = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed_layer(x)
        x = self.mixer_layers(x)
        x = self.neck(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    convmixer = ConvMixer(dim=1024, depth=8, patch_size=4, kernel_size=7, n_classes=10)
    summary(convmixer, (3,32,32), batch_size=256, device="cpu")
    
    




