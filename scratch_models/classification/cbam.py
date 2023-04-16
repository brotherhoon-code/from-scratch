import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


'''
ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/senet.py

## 1. 모델 설계  
ResNet에서 추가된 블록만 설명합니다.

SpatialAttentionBlock
    spatial 방향의 attn map을 구하는 블록입니다.

ChannelAttentionBlock
    channel 방향의  attn map을 구하는 블록입니다.


## 2. 구현 포인트  
einops에서 제공하는 layer를 이용했습니다.


## 3. 주의 사항  
32x32 이미지에서는 stage4에서 rank가 달라질 수 있어
dim을 고려하는 로직을 추가하였습니다.
    spatial_attn_map = self.spatial_attn_block(x)
    if spatial_attn_map.dim() == 1: 
        spatial_attn_map=repeat(spatial_attn_map, 'b -> b c h w', c=x.size(1), h=x.size(2), w=x.size(3))
    else:
        spatial_attn_map=repeat(spatial_attn_map, 'b h w -> b c h w', c=x.size(1))

'''


class SpatialAttentionBlock(nn.Module):
    def __init__(self, 
                 kernel_size=7, 
                 activation_func='sigmoid'):
        super(SpatialAttentionBlock, self).__init__()
        self.maxpool_layer = Reduce('b c h w -> b 1 h w', 'max')
        self.avgpool_layer = Reduce('b c h w -> b 1 h w', 'mean')
        self.conv_layer = nn.Conv2d(2, 1, kernel_size, padding='same')
        if activation_func == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            self.activation_func = nn.Sequential(nn.Tanh(), nn.ReLU())
    
    def forward(self, x):
        out1 = self.maxpool_layer(x)
        out2 = self.avgpool_layer(x)        
        out = torch.concat([out1, out2], dim=1)
        out = self.conv_layer(out)
        attn_map = self.activation_func(out)
        return attn_map.squeeze() # b h w
    

class ChannelAttentionBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 r=16, 
                 activation_func='sigmoid'):
        super(ChannelAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.r = r
        self.hidden_channels = in_channels//r
        
        if activation_func == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            self.activation_func = nn.Sequential(nn.Tanh(), nn.ReLU())
            
        self.maxpool_layer = Reduce('b c h w -> b c 1', 'max')
        self.avgpool_layer = Reduce('b c h w -> b c 1', 'mean')
        
        self.expansion_layer = nn.Linear(in_channels, self.hidden_channels)
        self.reduction_layer = nn.Linear(self.hidden_channels, in_channels)
        
        
    def forward(self, x):
        out1 = self.maxpool_layer(x).squeeze() # b c
        out2 = self.avgpool_layer(x).squeeze() # b c
        
        out1 = self.expansion_layer(out1)
        out1 = self.reduction_layer(out1)
        out2 = self.expansion_layer(out2)
        out2 = self.reduction_layer(out2)
        
        attn_map = self.activation_func(out1 + out2)
        return attn_map # b c


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_channels, 
                 channels, 
                 stride=1, 
                 kernel_size = 7,
                 activation_func = 'sigmoid',
                 groups=1, 
                 width_per_group=64,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_groups = width_per_group
        
        self.spatial_attn_block = SpatialAttentionBlock(kernel_size=kernel_size, activation_func=activation_func)
        self.channel_attn_block = ChannelAttentionBlock(channels*self.expansion,r=16, activation_func=activation_func)
        
        if groups != 1 or width_per_group != 64:
            raise ValueError("Basic only supperts groups=1 and base_width=64")
        self.width = int(channels*(width_per_group/64.))*groups
        
        self.shortcut=[]
        
        # 각 스테이지의 첫번째 블록인 경우 skip conn을 위해 차원을 확장시킵니다.
        if self.stride!=1 or self.in_channels != self.channels*self.expansion:
            self.shortcut.append(nn.Conv2d(self.in_channels, 
                                           self.channels*self.expansion, 
                                           kernel_size=(1,1),
                                           stride=self.stride))
            self.shortcut.append(nn.BatchNorm2d(self.channels*self.expansion))
            
        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(3,3), stride=self.stride, padding=1), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.width, out_channels=self.channels*self.expansion, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(self.channels*self.expansion), 
            # output channel dim is self.channels*1
        )
        
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block(x) # b c h w
        # b: x.size(0), c: x.size(1), h: x.size(2), w: x.size(3)
        
        channel_attn_map=repeat(self.channel_attn_block(x), 'b c -> b c h w', h=x.size(2), w=x.size(3))
        x = x*channel_attn_map
        
        spatial_attn_map = self.spatial_attn_block(x)
        
        if spatial_attn_map.dim() == 1: # e.g [2, 1, 1, 1]이나 squeeze에 의해 [2]로 표시됨
            spatial_attn_map=repeat(spatial_attn_map, 'b -> b c h w', c=x.size(1), h=x.size(2), w=x.size(3))
        else:
            spatial_attn_map=repeat(spatial_attn_map, 'b h w -> b c h w', c=x.size(1))
        
        x = x*spatial_attn_map
        out = x + identity
        return F.relu(out)


    

# 각 stage당 n번 반복되는 resnet bottleneck 블록입니다.
# renet50 이후로는 이 블록을 사용합니다.
class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, 
                 in_channels, 
                 channels, 
                 stride=1, 
                 kernel_size = 7,
                 activation_func = 'sigmoid',
                 groups=1, 
                 width_per_group=64,
                 **kwargs):
        super(BottleNeckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        
        self.spatial_attn_block = SpatialAttentionBlock(kernel_size=kernel_size, activation_func=activation_func)
        self.channel_attn_block = ChannelAttentionBlock(channels*self.expansion,r=16, activation_func=activation_func)
        
        
        # width: 1x1에 의해서 축소될 체널을 의미하는 변수입니다입니다.
        # 최종적으로는 self.channels*self.expansion dim으로 out됩니다.
        self.width = int(channels * (width_per_group/64.)) * groups
        self.shortcut = []
        
        if stride!=1 or in_channels!=channels*self.expansion:
            self.shortcut.append(nn.Conv2d(self.in_channels,
                                           self.channels*self.expansion, 
                                           kernel_size=(1,1),
                                           stride=self.stride))
            self.shortcut.append(nn.BatchNorm2d(self.channels*self.expansion))
        
        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            # contraction 1x1 conv
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            # 3x3 padding=1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(3,3), stride=self.stride, groups=self.groups, padding=1), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            # expasion 1x1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.channels*self.expansion, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.channels*self.expansion), 
            # output channel dim is self.channels*expansion
        )
        
        
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block(x) # b c h w
        # b: x.size(0), c: x.size(1), h: x.size(2), w: x.size(3)
        
        channel_attn_map=repeat(self.channel_attn_block(x), 'b c -> b c h w', h=x.size(2), w=x.size(3))
        x = x*channel_attn_map
        
        spatial_attn_map = self.spatial_attn_block(x)
        
        if spatial_attn_map.dim() == 1: # e.g [2, 1, 1, 1]이나 squeeze에 의해 [2]로 표시됨
            spatial_attn_map=repeat(spatial_attn_map, 'b -> b c h w', c=x.size(1), h=x.size(2), w=x.size(3))
        else:
            spatial_attn_map=repeat(spatial_attn_map, 'b h w -> b c h w', c=x.size(1))
        
        x = x*spatial_attn_map
        out = x + identity
        return F.relu(out)
    


class CBAM(nn.Module):
    def __init__(self, 
                 block_cls, 
                 n_blocks, 
                 n_classes, 
                 activation_func = 'sigmoid',
                 **kwargs):
        super(CBAM, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion
        
        self.stem = []
        self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3))
        self.stem.append(nn.BatchNorm2d(64))
        self.stem.append(nn.ReLU())
        self.stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stem_block = nn.Sequential(*self.stem)
        
        self.stage1_block = self._make_stage(self.block_cls, 64, 64, n_blocks[0], 1, activation_func)
        self.stage2_block = self._make_stage(self.block_cls, 64*self.expansion, 128, n_blocks[1], 2, activation_func)
        self.stage3_block = self._make_stage(self.block_cls, 128*self.expansion, 256, n_blocks[2], 2, activation_func)
        self.stage4_block = self._make_stage(self.block_cls, 256*self.expansion, 512, n_blocks[3], 2, activation_func)
        self.neck = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*self.expansion, self.n_classes)
        
        
    def forward(self, x):
        x = self.stem_block(x)
        x = self.stage1_block(x)
        x = self.stage2_block(x)
        x = self.stage3_block(x)
        x = self.stage4_block(x)
        x = self.neck(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def _make_stage(block, in_channels, out_channels, num_blocks, stride, activation_func = 'sigmoid'):
        stride_arr = [stride] + [1] * (num_blocks-1) # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        for stride in stride_arr:
            layers.append(block(in_channels=channels, channels=out_channels, stride=stride, activation_func=activation_func))
            channels = out_channels * block.expansion
        return nn.Sequential(*layers)


if __name__ == "__main__":
    
    m = CBAM(BottleNeckBlock, [3,4,6,3], n_classes=10)
    summary(m, (3,32,32), device='cpu')



    