import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


"""
ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/senet.py

## 1. 모델 설계  
레즈넷 기본 베이스에 identity map이 더해기지 직전 SEblock이 추가되는 구조입니다.

SEblock
    squeeze_block, excitation_block으로 구성되어 있습니다.
    차원을 맞춰주기 위해 사이에 flatten_layer를 추가했습니다.
    channel prob로 만드는 부분의 활성화 함수를 tanh + relu와 sigmoid 중 선택가능하도록 하였습니다.


## 2. 구현 포인트  
einops의 repeat를 이용해 차원을 변경하는 부분은 특이하니 익숙해질 필요가 있습니다.
    seblock_out = repeat(seblock_out, 'b c -> b c h w', h=x.size(2), w=x.size(3))
"""


# SEBlock은 b c h w 피처맵을 받아 h c 차원의 output을 내보냅니다
class SEblock(nn.Module):
    def __init__(self, in_channels, r=16, activation_func="sigmoid"):
        super(SEblock, self).__init__()
        self.in_channels = in_channels
        self.r = r
        self.hidden_channels = in_channels // r

        if activation_func == "sigmoid":
            self.activation_func = nn.Sigmoid()
        else:
            self.activation_func = nn.Sequential(nn.Tanh(), nn.ReLU())

        self.squeeze_block = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = nn.Flatten()
        self.excitation_block = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.in_channels),
        )

    def forward(self, x):
        x = self.squeeze_block(x)
        x = self.flatten_layer(x)
        x = self.excitation_block(x)
        x = self.activation_func(x)
        return x  # b c


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        activation_func="sigmoid",
        groups=1,
        width_per_group=64,
        **kwargs
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_groups = width_per_group
        self.seblock = SEblock(
            channels * self.expansion, r=16, activation_func=activation_func
        )

        if groups != 1 or width_per_group != 64:
            raise ValueError("Basic only supperts groups=1 and base_width=64")
        self.width = int(channels * (width_per_group / 64.0)) * groups

        self.shortcut = []

        # 각 스테이지의 첫번째 블록인 경우 skip conn을 위해 차원을 확장시킵니다.
        if self.stride != 1 or self.in_channels != self.channels * self.expansion:
            self.shortcut.append(
                nn.Conv2d(
                    self.in_channels,
                    self.channels * self.expansion,
                    kernel_size=(1, 1),
                    stride=self.stride,
                )
            )
            self.shortcut.append(nn.BatchNorm2d(self.channels * self.expansion))

        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.width,
                kernel_size=(3, 3),
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(self.width),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.width,
                out_channels=self.channels * self.expansion,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(self.channels * self.expansion),
            # output channel dim is self.channels*1
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block(x)  # b c h w
        seblock_out = self.seblock(x)
        seblock_out = repeat(seblock_out, "b c -> b c h w", h=x.size(2), w=x.size(3))
        x = x * seblock_out + identity
        return F.relu(x)


# 각 stage당 n번 반복되는 resnet bottleneck 블록입니다.
# renet50 이후로는 이 블록을 사용합니다.
class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride=1,
        activation_func="sigmoid",
        groups=1,
        width_per_group=64,
        **kwagrs
    ):
        super(BottleNeckBlock, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        self.seblock = SEblock(
            channels * self.expansion, r=16, activation_func=activation_func
        )

        # width: 1x1에 의해서 축소될 체널입니다.
        # 최종적으로는 self.channels*self.expansion dim으로 out됩니다.
        self.width = int(channels * (width_per_group / 64.0)) * groups
        self.shortcut = []

        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(
                nn.Conv2d(
                    self.in_channels,
                    self.channels * self.expansion,
                    kernel_size=(1, 1),
                    stride=self.stride,
                )
            )
            self.shortcut.append(nn.BatchNorm2d(self.channels * self.expansion))

        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            # contraction 1x1 conv
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.width,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(self.width),
            nn.ReLU(),
            # 3x3 padding=1 conv
            nn.Conv2d(
                in_channels=self.width,
                out_channels=self.width,
                kernel_size=(3, 3),
                stride=self.stride,
                groups=self.groups,
                padding=1,
            ),
            nn.BatchNorm2d(self.width),
            nn.ReLU(),
            # expasion 1x1 conv
            nn.Conv2d(
                in_channels=self.width,
                out_channels=self.channels * self.expansion,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(self.channels * self.expansion),
            # output channel dim is self.channels*expansion
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block(x)  # b c h w
        seblock_out = self.seblock(x)
        seblock_out = repeat(seblock_out, "b c -> b c h w", h=x.size(2), w=x.size(3))
        x = x * seblock_out + identity
        return F.relu(x)


class SEResNet(nn.Module):
    def __init__(
        self,
        block_cls,
        deep_stem,
        n_blocks,
        n_classes,
        activation_func="sigmoid",
        **kwargs
    ):
        super(SEResNet, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion

        self.stem = []
        if deep_stem:
            self.stem.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=(7, 7),
                    stride=2,
                    padding=3,
                )
            )
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())
            self.stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                )
            )
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())

        self.stem_block = nn.Sequential(*self.stem)

        self.stage1_block = self._make_stage(
            self.block_cls, 64, 64, n_blocks[0], 1, activation_func
        )
        self.stage2_block = self._make_stage(
            self.block_cls, 64 * self.expansion, 128, n_blocks[1], 2, activation_func
        )
        self.stage3_block = self._make_stage(
            self.block_cls, 128 * self.expansion, 256, n_blocks[2], 2, activation_func
        )
        self.stage4_block = self._make_stage(
            self.block_cls, 256 * self.expansion, 512, n_blocks[3], 2, activation_func
        )
        self.neck = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * self.expansion, self.n_classes)

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
    def _make_stage(
        block, in_channels, out_channels, num_blocks, stride, activation_func="sigmoid"
    ):
        stride_arr = [stride] + [1] * (num_blocks - 1)  # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        for stride in stride_arr:
            layers.append(
                block(
                    in_channels=channels,
                    channels=out_channels,
                    stride=stride,
                    activation_func=activation_func,
                )
            )
            channels = out_channels * block.expansion
        return nn.Sequential(*layers)


if __name__ == "__main__":
    m = SEResNet(BottleNeckBlock, False, [3, 4, 6, 3], 100, activation_func="sigmoid")
    summary(m, (3, 32, 32), device="cpu", batch_size=128)
