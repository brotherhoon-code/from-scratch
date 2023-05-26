import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channels=16, temperature=1.0):
        super(Attention, self).__init__()
        attention_channels = max(
            int(in_channels * reduction), min_channels
        )  # reduction과 min중에서 큰값을 attention 체널로 정의
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(attention_channels)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(
            in_channels=attention_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
        )
        self.func_channel = self.get_channel_attention  # pass

        if (
            in_channels == groups and in_channels == out_channels
        ):  # if depthwise not using func filter
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # if point-wise not using func_spatial
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=kernel_size * kernel_size,
                kernel_size=1,
                bias=True,
            )
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=kernel_num,
                kernel_size=1,
                bias=True,
            )
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x: torch.Tensor):
        channel_attention = torch.sigmoid(
            self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature
        )
        return channel_attention

    def get_filter_attention(self, x: torch.Tensor):
        filter_attention = torch.sigmoid(
            self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature
        )
        return filter_attention

    def get_spatial_attention(self, x: torch.Tensor):
        spatial_attention = self.spatial_fc(x).view(
            x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size
        )
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x: torch.Tensor):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return (
            self.func_channel(x),
            self.func_filter(x),
            self.func_spatial(x),
            self.func_kernel(x),
        )


class ODconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, reduction=0.0625, kernel_num=4, temperature=1.0):
        super(ODconv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        self.attention = Attention(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   groups=groups, reduction=reduction, kernel_num=kernel_num, temperature=temperature)
        
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        
        self._initialize_weights()
        
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
        
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature=temperature)
        
    def _forward_impl_common(self, x:torch.Tensor):
        channel_attn, filter_attn, spatial_attn, kernel_attn = self.attention(x)
        B, C, H, W = x.size()
        x = x*channel_attn
        x = x.reshape(1, -1, H, W)
        aggregate_weight = spatial_attn * kernel_attn * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_channels//self.groups, self.kernel_size, self.kernel_size]
        )
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups*B)
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        output = output*filter_attn
        return output
    
    def _forward_impl_pw1x(self, x):
        channel_attn, filter_attn, spatial_attn, kernel_attn = self.attention(x)
        x = x*channel_attn
        output =F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride,
                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = output*filter_attn
        return output
    
    def forward(self, x):
        return self._forward_impl(x)


if __name__ == "__main__":
    m = ODconv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0)
    out = m(torch.Tensor(2,64,16,16))
    print(out.shape)
    
    m = ODconv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    out = m(torch.Tensor(2,64,16,16))
    print(out.shape)
    print("hello world!")
