import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg16

def decomp_vgg16(use_drop=False):
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = model.classifier
    
    classifier = list(classifier)
    
    del classifier[6]
    if not use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    
    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad=False
            
    return nn.Sequential(*features), classifier