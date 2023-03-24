from functools import partial
import torch
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import torch.nn as nn

'''
reference
    https://github.com/rishikksh20/MLP-Mixer-pytorch
'''

class PatchEmbedBlock(nn.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 feature_dim):
        super().__init__()
        assert img_size%patch_size==0, 'img_size%patch_size != 0'
        self.img_size = img_size
        self.n_patches = (img_size%patch_size)**2
        self.embed_layer = nn.Conv2d(3, 
                                     feature_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        self.flatten_layer = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        assert x.shape[2] == self.img_size, 'input height != img_size'
        assert x.shape[3] == self.img_size, 'input width != img_size'
        x = self.embed_layer(x)
        x = self.flatten_layer(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, dim, expansion_ratio, dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(expansion_ratio*dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 feature_dim,
                 expansion_ratio):
        super().__init__()
        self.n_patches = (img_size//patch_size)**2
        self.feature_dim = feature_dim
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.transpose_layer = Rearrange('b p c -> b c p')
        self.token_mix_layer = DenseBlock(dim=self.n_patches, 
                                          expansion_ratio=expansion_ratio)
        self.channel_mix_layer = DenseBlock(dim=feature_dim, 
                                            expansion_ratio=expansion_ratio)
        
    def forward(self, x):
        x1 = self.layer_norm(x)
        x1 = self.transpose_layer(x1)
        x1 = self.token_mix_layer(x1)
        x1 = self.transpose_layer(x1)
        x = x + x1
        x2 = self.layer_norm(x)
        x2 = self.channel_mix_layer(x2)
        x = x + x2
        return x


class GlobalAvgPool1d(nn.Module):
    def forward(self, x):
        # Compute global average pooling along the tokens dimension
        x_pool = torch.mean(x, dim=1)
        return x_pool


class MLP_Mixer(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 feature_dim,
                 expansion_ratio,
                 depth,
                 n_classes
                 ):
        super().__init__()
        self.n_patches = (img_size//patch_size)**2
        self.embed_block = PatchEmbedBlock(img_size, 
                                           patch_size, 
                                           feature_dim)
        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(img_size, 
                         patch_size, 
                         feature_dim, 
                         expansion_ratio) for i in range(depth)],
            nn.LayerNorm(feature_dim),
            GlobalAvgPool1d()
        )
        self.classifier = nn.Linear(feature_dim, 
                                    n_classes)

    def forward(self, x):
        x = self.embed_block(x) # b p d
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    mixer = MLP_Mixer(img_size = 32, 
                      patch_size = 4, 
                      feature_dim = 128, 
                      expansion_ratio = 2, 
                      depth = 8,
                      n_classes = 10)
    
    summary(mixer, (3,32,32), device='cpu')