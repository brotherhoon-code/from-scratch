import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from math import sqrt

'''
reference
    http://einops.rocks/pytorch-examples.html
    boostcamp_aitech_homework
     
## 1. 모델 설계  
계층 구조를 아래와 같이 설계하였습니다.  
```
VisionTransformer
    PatchEmbeddingBlock
    EncoderBlock
        MultiHeadSelfAttentionBlock
        MlpBlock
    Classifier
```
  
## 2. 구현 포인트  
* N개의 qkv를 생성하는 linear layer는 N의 개수와 무관하게 qkv당 각 1개씩만 존재합니다.  
* Encoder block의 skip conn에는 dropout이 존재합니다.  
* attention의 activation function은 MLP에 단 한개밖에 없습니다.  
* classifier의 feature는 패치 average가 아니라 embed_dim의 average입니다.  
* eniops를 적극적으로 사용하였습니다.  
  
## 3. 주의 사항  
* 포지셔널 인코딩을 learnable random params로 변경하였습니다.  
* 클래스 토큰을 learnable random params로 변경하였습니다.  
* 클래스 토큰을 classification의 피처로 이용하지 않고,  
 embed_dim의 avg를 cls의 피처로 이용했습니다.


'''

class PatchEmbeddingBlock(nn.Module):
    def __init__(self, 
                 img_channel:int, 
                 img_height:int, 
                 img_width:int, 
                 patch_size:int, 
                 emb_dim: int
                 ):
        super(PatchEmbeddingBlock, self).__init__()
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_patch_height = self.img_height//self.patch_size
        self.n_patch_width = self.img_width//self.patch_size
        self.n_patches = self.n_patch_height * self.n_patch_width
        
        self.cls_token = nn.Parameter(torch.randn(1, 1,self.emb_dim))
        self.pos = nn.Parameter(torch.randn(self.n_patches+1, emb_dim))
        
        self.flatten_layer = Rearrange('b c (p_h h_n) (p_w w_n) -> b (h_n w_n) (c p_h p_w)', 
                                       p_h = self.patch_size, 
                                       h_n = self.n_patch_height,
                                       p_w = self.patch_size, 
                                       w_n = self.n_patch_width)
        self.linear_proj_layer = nn.Linear(self.patch_size*self.patch_size*self.img_channel, self.emb_dim)
    
    def forward(self, x):
        n_batch, _, _, _ = x.shape
        x = self.flatten_layer(x)
        x = self.linear_proj_layer(x)
        token = repeat(self.cls_token,'() n d -> b n d', b=n_batch)
        x = torch.cat((token, x), dim=1)
        x += self.pos
        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, 
                 emb_dim:int, 
                 qkv_dim:int,
                 num_heads:int=8, 
                 dropout_ratio:float=0.2):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.emb_dim = emb_dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.full_dims = self.qkv_dim * self.num_heads
        self.root_key_dim = sqrt(self.qkv_dim)
        
        self.query_layer = nn.Linear(emb_dim, self.full_dims)
        self.key_layer = nn.Linear(emb_dim, self.full_dims)
        self.value_layer = nn.Linear(emb_dim, self.full_dims)
        
        self.dropout_layer = nn.Dropout(dropout_ratio)
        self.rearrange_to_multi_head = Rearrange('b n (h d) -> b h n d', h=self.num_heads)
        self.transpose_key = Rearrange('b h n d -> b h d n')
        self.rearrange_to_single_head = Rearrange('b h n d -> b n (h d)', h=self.num_heads)
        self.linear = nn.Linear(self.qkv_dim*self.num_heads, self.emb_dim)
    
    def forward(self, x):
        full_query = self.query_layer(x)
        full_key = self.key_layer(x)
        full_value = self.value_layer(x)
        
        multi_query = self.rearrange_to_multi_head(full_query)
        multi_key = self.rearrange_to_multi_head(full_key)
        multi_value = self.rearrange_to_multi_head(full_value)
        
        transposed_multi_key = self.transpose_key(multi_key)
        scailed_attn_score = torch.matmul(multi_query,transposed_multi_key)/self.root_key_dim
        attn_map = torch.softmax(scailed_attn_score, dim=-1)
        attn_map = self.dropout_layer(attn_map) # after softmax, dropout
        
        multihead_features = torch.matmul(attn_map, multi_value)
        
        singlehead_features = self.rearrange_to_single_head(multihead_features) # b h n d -> b n (h d)
        multihead_features = self.linear(singlehead_features) # b n (h d) --> b n d
        
        return multihead_features # attn_map


class MlpBlock(nn.Module):
    def __init__(self, 
                 emb_dim:int, 
                 forward_dim:int, 
                 dropout_ratio:float=0.2):
        super(MlpBlock, self).__init__()
        self.linear_1 = nn.Linear(emb_dim, forward_dim)
        self.dropout_layer = nn.Dropout(dropout_ratio)
        self.linear_2 = nn.Linear(forward_dim, emb_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.dropout_layer(x) # after gelu, dropout
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim:int, 
                 qkv_dim:int,
                 n_heads:int, 
                 hidden_dim:int, 
                 dropout_ratio:float):
        super(EncoderBlock, self).__init__()
        self.attention_layer = MultiHeadSelfAttentionBlock(emb_dim, qkv_dim, n_heads, dropout_ratio)
        self.mlp_layer = MlpBlock(emb_dim, hidden_dim, dropout_ratio)
        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.residual_dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x):
        x_ = self.layer_norm_1(x)
        x_ = self.attention_layer(x_)
        x = x_ + self.residual_dropout(x)
        x_ = self.layer_norm_2(x)
        x_ = self.mlp_layer(x)
        x = x_ + self.residual_dropout(x)
        return x # , attn_map


class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_channel:int, 
                 img_height:int, 
                 img_width:int, 
                 patch_size:int, 
                 emb_dim:int, 
                 qkv_dim:int,
                 n_heads:int, 
                 dropout_ratio:float, 
                 hidden_dim:int, 
                 n_encoders:int, 
                 n_classes:int):
        super(VisionTransformer, self).__init__()
        self.patch_embed_layer = PatchEmbeddingBlock(img_channel, 
                                                     img_height, 
                                                     img_width, 
                                                     patch_size, 
                                                     emb_dim)
        encoder_list = [EncoderBlock(emb_dim, qkv_dim, n_heads, hidden_dim, dropout_ratio) for i in range(n_encoders)]
        self.encoder_layers = nn.Sequential(*encoder_list)
        self.reduce_layer = Reduce('b n d -> b d', reduction='mean')
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)
        
    def forward(self, x):
        x = self.patch_embed_layer(x)
        x = self.encoder_layers(x)
        x = self.reduce_layer(x)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    img = torch.randn(256,3,32,32)
    vit = VisionTransformer(img_channel=3, 
                            img_height=32, 
                            img_width=32, 
                            patch_size=4, 
                            emb_dim=128, 
                            qkv_dim=64,
                            n_heads=8,
                            dropout_ratio=0.2,
                            hidden_dim=128, 
                            n_encoders=8, 
                            n_classes=10)
    
    output = vit(img)
    summary(vit, (3,32,32), device='cpu')