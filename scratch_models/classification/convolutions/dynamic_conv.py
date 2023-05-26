import torch
import torch.nn as nn
import torch.nn.functional as F


# input (B, C, H, W)
# output (B, K)
class attention2d(nn.Module):
    def __init__(self, in_channels, ratio, K, temperature, init_weight=True):
        super().__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels != 3:  # not RGB
            hidden_channels = int(in_channels * ratio) + 1
        else:
            hidden_channels = K
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=K, kernel_size=(1, 1), bias=True
        )
        self.temperature = temperature
        if init_weight:  # default weight update
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print(f"change temperature to: {self.temperature}")

    def forward(self, x):
        x = self.avgpool(x)  # [B, in_channels, 1, 1]
        x = self.fc1(x)  # [B, in_channels*ratio+1, 1, 1]
        x = F.relu(x)
        x = self.fc2(x)  # [B, K, 1, 1]
        x = x.view(x.size(0), -1)  # [B, K]
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=34,
        init_weight=True,
    ):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(
            in_channels=in_channels, ratio=ratio, K=K, temperature=temperature
        )

        # [K, out_c, in_c, k, k]
        self.weight = nn.Parameter(
            torch.randn(
                K, out_channels, in_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        attention_score = self.attention(x)
        B, C, H, W = x.size()
        x = x.view(1, -1, H, W)  # [1, B*C, H, W]
        weight = self.weight.view(self.K, -1)  # [K, out_c*in_c*k*k]

        aggregate_weight = torch.mm(attention_score, weight).view(
            B * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        if self.bias is not None:
            print(self.bias.view(-1).shape)
            aggregate_bias = torch.mm(attention_score, self.bias).view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        return output


if __name__ == "__main__":
    m = Dynamic_conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        ratio=0.25,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=40,
    )
    output = m(torch.Tensor(4, 64, 224, 224))
    print(output.shape)
