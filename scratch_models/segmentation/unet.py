import torch
import torch.nn as nn
from torchsummary import summary

class UNet(nn.Module):
    
    def _CBR(self,
             in_channels,
             out_channels,
             kernel_size=3,
             stride=1,
             padding=1,
             bias=True):
        cbr = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        return cbr
    
    def __init__(self, num_classes=11):
        super().__init__()
        
        # contracting path
        self.enc1_1 = self._CBR(3, 64)
        self.enc1_2 = self._CBR(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = self._CBR(64, 128)
        self.enc2_2 = self._CBR(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3_1 = self._CBR(128, 256)
        self.enc3_2 = self._CBR(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = self._CBR(256, 512)
        self.enc4_2 = self._CBR(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = self._CBR(512, 1024)
        self.enc5_2 = self._CBR(1024, 1024)
        self.unpool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec4_2 = self._CBR(1024,512)
        self.dec4_1 = self._CBR(512,512)
        self.unpool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = self._CBR(512,256)
        self.dec3_1 = self._CBR(256,256)
        self.unpool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        
        # concated channels in
        self.dec2_2 = self._CBR(256,128)
        self.dec2_1 = self._CBR(128,64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=True)
        
        # concated channels in
        self.dec1_2 = self._CBR(128,64)
        self.dec1_1 = self._CBR(64, 64)
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)
        
        unpool4 = self.unpool4(enc5_2)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # concat
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1) # concat
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1) # concat
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1) # concat
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        output = self.score_fr(dec1_1)
        
        return output
    
if __name__ == "__main__":
    m = UNet(10)
    summary(m, (3, 224,224), batch_size=1, device='cpu')
    
        
        
        