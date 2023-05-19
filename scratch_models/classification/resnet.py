import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


"""
ref: https://github.com/xxxnell/spatial-smoothing/blob/master/models/resnet.py

## 1. 모델 설계  
ResNet
    stem_block
        __init__에서 nn.Sequential을 통해서 만들어집니다.
        stride와 maxpool을 이용하여 resolution의 축소가 일어납니다.
    make_stage
        __init__에 argument로 전달된 nn.Module (e.g)BasicBlock, BottleNeckBlock의 반복으로 레이어를 만듭니다.
        2번쨰 스테이부터 첫번째 block에서만 stride가 존재하므로 이를 반영했습니다.
    neck
        Global Average Pooling입니다.
    classifier
        nn.Linear입니다.
        
BasicBlock
    resnet18과 resnet34에서 사용하는 블록입니다.
    기본 블록는 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU로 설계되어있습니다.
    stride !=1 아닐 경우 차원수가 맞지 않아 identity를 더할 수 없으므로 stride 1x1 Conv를 이용하여 resolution을 줄입니다.

BottleNeckBlock
    resnet50이후에서 사용하는 블록입니다.
    기본 블록은 1x1 conv(체널 축소) -> BN,ReLU -> 3x3 conv -> BN,ReLU -> 1x1 conv(체널확장) -> BN, ReLU로 설계되어있습니다.
    그외 stride != 1을 고려한 경우는 BasicBlock과 동일합니다.


## 2. 구현 포인트  
make_stage를 이용하여 반복적인 구조 설계가 포인트 입니다.
첫번째 stage의 block에만 적용되는 stride를 어떻게 처리하는 로직이 재미있습니다.
    stride_arr = [stride] + [1] * (num_blocks-1)


## 3. 주의 사항  
레즈넷은 봐도봐도 복잡합니다.
"""


# 각 stage당 n번 반복되는 resnet 기본 블록입니다.
# renet18과 resnet34는 이 블록을 사용합니다.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, channels, stride=1, groups=1, width_per_group=64, **kwargs
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_groups = width_per_group

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
        skip = self.shortcut(x)
        x = self.block(x) + skip
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
        skip = self.shortcut(x)
        x = self.block(x) + skip
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(
        self,
        block_cls=BottleNeckBlock,
        deep_stem=True,  # for cifar
        n_blocks=[3, 4, 6, 3],
        n_classes=100,
        **kwargs
    ):
        super(ResNet, self).__init__()
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
        self.stage1_block = self._make_stage(self.block_cls, 64, 64, n_blocks[0], 1)
        self.stage2_block = self._make_stage(
            self.block_cls, 64 * self.expansion, 128, n_blocks[1], 2
        )
        self.stage3_block = self._make_stage(
            self.block_cls, 128 * self.expansion, 256, n_blocks[2], 2
        )
        self.stage4_block = self._make_stage(
            self.block_cls, 256 * self.expansion, 512, n_blocks[3], 2
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
    def _make_stage(block, in_channels, out_channels, num_blocks, stride):
        stride_arr = [stride] + [1] * (num_blocks - 1)  # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        for stride in stride_arr:
            layers.append(block(channels, out_channels, stride))
            channels = out_channels * block.expansion
        return nn.Sequential(*layers)


if __name__ == "__main__":
    m = ResNet(BottleNeckBlock, True, [3, 4, 6, 3], 10)
    summary(m, (3, 224, 224), device="cpu", batch_size=128)
