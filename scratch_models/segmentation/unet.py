import torch
import torch.nn as nn
from torchsummary import summary

"""
논문과는 다르게 크롭 부분 삭제함.
"""


class CBR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.cbr1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.cbr2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.enc1 = CBR_Block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = CBR_Block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = CBR_Block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = CBR_Block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5 = CBR_Block(512, 1024)
        self.unpool4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2, padding=0, bias=True
        )

        self.dec4 = CBR_Block(1024, 512)  # 512+512 -> 512
        self.unpool3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2, padding=0, bias=True
        )  # 512 -> 256

        self.dec3 = CBR_Block(256 * 2, 256)  # 256+256 -> 256
        self.unpool2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, padding=0, bias=True
        )  # 256 -> 128

        self.dec2 = CBR_Block(128 * 2, 128)  # 128+128 -> 128
        self.unpool1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, padding=0, bias=True
        )  # 128 -> 64

        self.dec1 = CBR_Block(64 * 2, 64)

        self.seg_head = nn.Conv2d(64, n_classes, 1, 1)

    def forward(self, x):
        enc1_out = self.enc1(x)  # 3 -> 64
        pool1_out = self.pool1(enc1_out)  # 64 -> 64

        enc2_out = self.enc2(pool1_out)  # 64 -> 128
        pool2_out = self.pool2(enc2_out)  # 128 -> 128

        enc3_out = self.enc3(pool2_out)  # 128 -> 256
        pool3_out = self.pool3(enc3_out)  # 256 -> 256

        enc4_out = self.enc4(pool3_out)  # 256 -> 512
        pool4_out = self.pool4(enc4_out)  # 512 -> 512

        enc5_out = self.enc5(pool4_out)  # 512 -> 1024

        out = self.unpool4(enc5_out)  # 1024 -> 512
        out = torch.cat((out, enc4_out), dim=1)  # concat 512+512=1024 -> 1024
        out = self.dec4(out)  # 1024 -> 512

        out = self.unpool3(out)  # 512-> 256
        out = torch.cat((out, enc3_out), dim=1)
        out = self.dec3(out)  # concat 256*2 -> 256

        out = self.unpool2(out)  # 256 -> 128
        out = torch.cat((out, enc2_out), dim=1)
        out = self.dec2(out)  # 128*2 -> 128

        out = self.unpool1(out)  # 128 -> 64
        out = torch.cat((out, enc1_out), dim=1)  # 64 -> 64*2
        out = self.dec1(out)  # 64*2 -> 64
        out = self.seg_head(out)  # 64 -> n_classes

        return out


if __name__ == "__main__":
    img = (3, 224, 224)
    m = UNet(10)
    summary(m, img, batch_size=1, device="cpu")
    # 31,044,106
