import torch
from torch import nn
from modules import BasicBlock, Bottleneck

'''
ref: https://github.com/stefanopini/simple-HRNet
'''


class StageModule(nn.Module):
    def __init__(self,
                 stage,
                 output_branches,
                 c,
                 bn_momentum):
        super().__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        
        for i in range(self.stage):
            w = c * (2**i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum)
            )
            self.branches.append(branch)
        
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c*(2**j), c*(2**i), kernel_size=(1,1), stride=(1,1), bias=False),
                        nn.BatchNorm2d(c*(2**i),eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0**(j-i)), mode='nearest')))
                elif i > j:
                    ops = []
                    for k in range(i-j-1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True)))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        
        self.relu = nn.ReLU(inplace=True)
                    
        
        
    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        
        return x_fused
    
    
class HRNet(nn.Module):
    def __init__(self,
                 c=48,
                 nof_joints=17,
                 bn_momentum=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        # stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        )
        
        self.layer1 = nn.Sequential(
            Bottleneck(64,64,downsample=downsample),
            Bottleneck(256,64),
            Bottleneck(256,64),
            Bottleneck(256,64),
        )
        
        # fusion layer1
        self.transition1 = nn.ModuleList([
            # stream1
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),
            # stream2
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, c*(2**1), kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
                    nn.BatchNorm2d(c*(2**1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)
                ),
            )
        ])
        
        # stage2
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum)
        )
        
        # fusion layer2
        self.transition2 = nn.ModuleList(
            [
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(nn.Sequential(
                    nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )),
            ]
        )
        
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )
        
        self.transition3 = nn.ModuleList(
            [
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(),
                nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
            ]
        )
        
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )
        
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1,1), stride=(1,1))
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]
        
        x = self.stage3(x)
        
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]
        
        x = self.stage4(x)
        x = self.final_layer(x[0])
        
        return x
    

if __name__=='__main__':
    m = HRNet(32,17,0.1)
    out = m(torch.ones(1,3,384,288))
    print(out.shape)