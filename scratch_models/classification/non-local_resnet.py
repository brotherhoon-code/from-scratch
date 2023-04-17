import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from typing import Callable

'''
ref: https://github.com/17Skye17/Non-local-Neural-Networks-Pytorch/tree/master/lib

'''
class GaussianNonLocalBlock(nn.Module):
    # 쿼리와 키를 임배딩하지 하지 않습니다.
    # 쿼리와 키는 flatten하고 dot product가 수행됩니다.
    # 밸류는 임배딩 됩니다.
    # W에 의해서 입력 체널로 복원됩니다.
    def __init__(self, 
                 in_channels:int,
                 r:int=2):
        super(GaussianNonLocalBlock, self).__init__()
        embed_dim = in_channels//r
        
        # theta
        self.q_embed_layer = nn.Sequential(Rearrange('b c h w -> b (h w) c ')) # B N C
        # phi
        self.k_embed_layer = nn.Sequential(Rearrange('b c h w -> b c (h w)')) # B C N
        # g
        self.v_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=embed_dim, 
                                                     kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c')) # B N c
        # W
        self.expansion_layer = nn.Conv2d(in_channels=embed_dim, 
                                         out_channels=in_channels, 
                                         kernel_size=(1,1))    # B C H W
    def forward(self, x):
        identity = x # B C H W
        _, _, h, w = x.size()
        q = self.q_embed_layer(x) # B N C
        k = self.k_embed_layer(x) # B C N
        v = self.v_embed_layer(x) # B N c
        attn_score = torch.matmul(q,k) # B N N
        attn_prob = F.softmax(attn_score, dim=-1) # B N N
        embed = torch.matmul(attn_prob, v) # B N c
        embed = rearrange(embed, 'b (h w) c -> b c h w', h=h, w=w) # B c H W
        embed = self.expansion_layer(embed) # B C H W
        x = embed + identity
        return x
    

class EmbeddedGaussianNonLocalBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 r:int=2):
        super(EmbeddedGaussianNonLocalBlock, self).__init__()
        embed_dim = int(in_channels/2)
        
        # theta
        self.q_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=embed_dim, 
                                                      kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c ')) # B N c
        # phi
        self.k_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=embed_dim, 
                                                     kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b c (h w)')) # B c N
        # g
        self.v_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=embed_dim, 
                                                     kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c')) # B N c
        # W
        self.expansion_layer = nn.Conv2d(in_channels=embed_dim, 
                                         out_channels=in_channels, 
                                         kernel_size=(1,1)) # B C H W
        
    def forward(self, x):
        identity = x # B C H W
        q = self.q_embed_layer(x) # B N c
        k = self.k_embed_layer(x) # B c N
        v = self.v_embed_layer(x) # B N c
        attn_score = torch.matmul(q, k) # B N N
        attn_prob = torch.softmax(attn_score, dim=-1) # B N N
        embedded_feature = torch.matmul(attn_prob, v) # B N c
        feature_map = rearrange(embedded_feature, 'b (h w) c -> b c h w ', h=x.size(2), w=x.size(3)) # B c H W
        x = self.expansion_layer(feature_map) + identity # B C H W
        return x


class DotProductNonLocalBlock(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 r:int=2):
        super(DotProductNonLocalBlock, self).__init__()
        embed_dim = int(in_channels/r)
        # theta
        self.q_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=embed_dim, 
                                                      kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c ')) # B N c
        # phi
        self.k_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=embed_dim, 
                                                     kernel_size=(1,1)),  
                                           Rearrange('b c h w -> b c (h w)')) # B c N
        # g
        self.v_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=embed_dim, 
                                                     kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c')) # B N c
        # W
        self.expansion_layer = nn.Conv2d(in_channels=embed_dim, 
                                         out_channels=in_channels, 
                                         kernel_size=(1,1)) # B C H W
        
    def forward(self, x):
        _, _, h, w = x.size()
        identity = x
        q = self.q_embed_layer(x) # B N c
        k = self.k_embed_layer(x) # B c N
        v = self.v_embed_layer(x) # B N c
        
        attn_score = torch.matmul(q, k) # B N N
        N = attn_score.size(-1) # B N N
        scaled_attn_score = attn_score/N # B N N
        
        embed = torch.matmul(scaled_attn_score, v) # B N c
        embed = rearrange(embed, 'b (h w) c -> b c h w', h=h, w=w) # B c H W
        embed = self.expansion_layer(embed) # B C H W
        x = embed + identity # B C H W
        return x


class ConcatNonLocalBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 r:int=2):
       super(ConcatNonLocalBlock, self).__init__()
       embed_dim = in_channels//r
        # theta
       self.q_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                    out_channels=embed_dim, 
                                                    kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b c (h w)')) # B c N
       # phi
       self.k_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                    out_channels=embed_dim, 
                                                    kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b c (h w)')) # B c N
       # g
       self.v_embed_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                    out_channels=embed_dim, 
                                                    kernel_size=(1,1)), 
                                           Rearrange('b c h w -> b (h w) c')) # B N c
       # concat_proj
       self.concat_project_layer = nn.Sequential(nn.Conv2d(in_channels=embed_dim*2, 
                                                           out_channels=1, 
                                                           kernel_size=(1,1), 
                                                           bias=False), 
                                                 nn.ReLU()) # B 2c H W -> B 1 H W
       # W
       self.expansion_layer = nn.Conv2d(in_channels=embed_dim, 
                                         out_channels=in_channels, 
                                         kernel_size=(1,1)) # B C H W
    
    def forward(self, x):
        identity = x
        _, _, f_h, f_w = x.size()
        q = self.q_embed_layer(x) # B c N
        k = self.k_embed_layer(x) # B c N
        v = self.v_embed_layer(x) # B N c
        q = repeat(q, 'b c n1 -> b c n1 n2', n2=q.size(-1)) # B c N N
        k = repeat(k, 'b c n1 -> b c n1 n2', n2=k.size(-1)) # B c N N
        qk = torch.concat([q,k], dim=1) # B 2c N N
        attn_score = self.concat_project_layer(qk) # B 1 N N
        attn_score = attn_score.squeeze() # B N N
        N = attn_score.size(-1) # N
        scaled_attn_score = attn_score/N # B N N
        
        if scaled_attn_score.dim() != 3: # 32x32 이미지에서 1x1 attn score의 경우 channel dim만 남게되므로 추가함
            scaled_attn_score = scaled_attn_score.unsqueeze(dim=-1)
            scaled_attn_score = scaled_attn_score.unsqueeze(dim=-1)
            
        embed = torch.matmul(scaled_attn_score, v) # B N c
        embed = rearrange(embed, 'b (h w) c -> b c h w', h=f_h, w=f_w) # B c H W
        embed = self.expansion_layer(embed) # B C H W
        x = embed + identity # B C H W
        return x


################################## Resnet과 동일합니다.

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
        x = F.relu(x)
        return x

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
        x = F.relu(x)
        return x
    

class NonLocalResNet(nn.Module):
    def __init__(self, 
                 block_cls = BottleNeckBlock, 
                 deep_stem = True,
                 n_blocks = [3,4,6,3],
                 n_classes = 100, 
                 n_local_cls = "EmbeddedGaussianNonLocalBlock", 
                 n_local_stage_idx = [1, 2, 3, 4],
                 **kwargs):
        super(NonLocalResNet, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion
        
        self.non_local_blocks = nn.ModuleList()
        if n_local_cls == "GaussianNonLocalBlock":
            n_local_cls = GaussianNonLocalBlock
        elif n_local_cls == "DotProductNonLocalBlock":
            n_local_cls = DotProductNonLocalBlock
        elif n_local_cls == "ConcatNonLocalBlock":
            n_local_cls = ConcatNonLocalBlock
        else:
            n_local_cls = EmbeddedGaussianNonLocalBlock
        
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
        self.non_local_blocks.append(n_local_cls(in_channels=64*self.expansion, r=2)) if 1 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        
        self.stage2_block = self._make_stage(self.block_cls, 64*self.expansion, 128, n_blocks[1], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=128*self.expansion, r=2)) if 2 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        
        self.stage3_block = self._make_stage(self.block_cls, 128*self.expansion, 256, n_blocks[2], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=256*self.expansion, r=2)) if 3 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())

        
        self.stage4_block = self._make_stage(self.block_cls, 256*self.expansion, 512, n_blocks[3], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=512*self.expansion, r=2)) if 4 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        
        self.neck = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*self.expansion, self.n_classes)
        
    def forward(self, x):
        x = self.stem_block(x)
        
        x = self.stage1_block(x)
        x = self.non_local_blocks[0](x)
        
        x = self.stage2_block(x)
        x = self.non_local_blocks[1](x)
        
        x = self.stage3_block(x)
        x = self.non_local_blocks[2](x)
        
        x = self.stage4_block(x)
        x = self.non_local_blocks[3](x)
        
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

if __name__ == "__main__":
    
    m = NonLocalResNet(block_cls=BottleNeckBlock, 
                       deep_stem = False,
                       n_blocks=[3,4,6,3], 
                       n_classes=100, 
                       n_local_cls="EmbeddedGaussianNonLocalBlock",
                       n_local_stage_idx=[1,2,3,4])
    
    summary(m, (3,32,32), batch_size=128, device="cpu")
    
    
    
    
    
    
    