import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, n_heads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(d_model = hidden_dim,
                                          n_heads = n_heads, 
                                          num_encoder_layers = num_encoder_layers, 
                                          num_decoder_layers = num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes+1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos =  nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        
    def forward(self, x):
        x = self.backbone(x)
        h:torch.Tensor = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos+h.flatten(2).permute(2,0,1),
                             self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()