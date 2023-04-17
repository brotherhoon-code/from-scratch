import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


'''
ref: https://github.com/dl19940602/GCnet-pytorch
'''

# 각 stage당 n번 반복되는 resnet 기본 블록입니다.
# renet18과 resnet34는 이 블록을 사용합니다.
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_channels, 
                 channels, 
                 stride=1, 
                 groups=1, 
                 width_per_group=64,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_groups = width_per_group
        
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
        skip = self.shortcut(x)
        x = self.block(x) + skip
        return F.relu(x)
    

# 각 stage당 n번 반복되는 resnet bottleneck 블록입니다.
# renet50 이후로는 이 블록을 사용합니다.
class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, 
                 in_channels:int, 
                 channels:int, 
                 stride=1, 
                 groups=1, 
                 width_per_group=64, 
                 **kwagrs):
        super(BottleNeckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        
        
        # width: 1x1에 의해서 축소될 체널입니다.
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
        skip = self.shortcut(x)
        x = self.block(x) + skip
        return F.relu(x)



class GCNet(nn.Module):
    def __init__(self, 
                 block_cls =  BottleNeckBlock, 
                 deep_stem = True, # for cifar
                 n_blocks = [3, 4, 6, 3],
                 n_classes = 100, 
                 n_context_stage_idx = [1, 2, 3, 4],
                 **kwargs):
        super(GCNet, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion
        
        self.context_blocks = nn.ModuleList()
        
        self.stem = []
        if deep_stem:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())
            self.stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())
        
        self.stem_block = nn.Sequential(*self.stem)
        
        self.stage1_block = self._make_stage(self.block_cls, 64, 64, n_blocks[0], 1)
        self.context_blocks.append(
            GlobalContextBlock(in_channels=64*self.expansion)) if 1 in n_context_stage_idx else self.context_blocks.append(nn.Sequential())
        
        self.stage2_block = self._make_stage(self.block_cls, 64*self.expansion, 128, n_blocks[1], 2)
        self.context_blocks.append(
            GlobalContextBlock(in_channels=128*self.expansion)) if 2 in n_context_stage_idx else self.context_blocks.append(nn.Sequential())
        
        self.stage3_block = self._make_stage(self.block_cls, 128*self.expansion, 256, n_blocks[2], 2)
        self.context_blocks.append(
            GlobalContextBlock(in_channels=256*self.expansion)) if 3 in n_context_stage_idx else self.context_blocks.append(nn.Sequential())
        
        self.stage4_block = self._make_stage(self.block_cls, 256*self.expansion, 512, n_blocks[3], 2)
        self.context_blocks.append(
            GlobalContextBlock(in_channels=512*self.expansion)) if 4 in n_context_stage_idx else self.context_blocks.append(nn.Sequential())
        
        self.neck = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*self.expansion, self.n_classes)
        
        
    def forward(self, x):
        x = self.stem_block(x)
        
        x = self.stage1_block(x)
        x = self.context_blocks[0](x)
        
        x = self.stage2_block(x)
        x = self.context_blocks[1](x)
        
        x = self.stage3_block(x)
        x = self.context_blocks[2](x)
        
        x = self.stage4_block(x)
        x = self.context_blocks[3](x)
        
        x = self.neck(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def _make_stage(block, in_channels, out_channels, num_blocks, stride):
        stride_arr = [stride] + [1] * (num_blocks-1) # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        for stride in stride_arr:
            layers.append(block(channels, out_channels, stride))
            channels = out_channels * block.expansion
        return nn.Sequential(*layers)


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 in_channels:int = 64, 
                 ratio:int = 16, 
                 pool:str = 'attn', 
                 fusion:str = 'add'):
        super(GlobalContextBlock, self).__init__()
        self.reshape_q = Rearrange('b c h w -> b c (h w)')
        
        self.W_k = nn.Conv2d(in_channels, 1, 1)
        self.reshape_k = Rearrange('b c h w -> b (h w) c')
        
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, 1),
            nn.LayerNorm([in_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, 1),
        )
        
    def forward(self, x):
        _, _, height, width = x.size(0), x.size(1), x.size(2), x.size(3)
        identity = x
        Q = self.reshape_q(x) # b c n
        Q = F.softmax(Q, dim=-1)
        
        K = self.W_k(x)
        K = self.reshape_k(K) # b n 1
        
        context = torch.matmul(Q, K).unsqueeze(-1) # b c 1 1
        context = self.transform(context) # b c 1 1
        context = repeat(context, 'b c 1 1-> b c h w', h=height, w=width) # b c h w
        
        out = context + identity # b c h w
        
        return out

if __name__ == "__main__":
    m = GCNet()
    summary(m, (3,224,224), device='cpu', batch_size=1)