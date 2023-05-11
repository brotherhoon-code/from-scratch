import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# make custom func
def get_resnet50_layer(model:nn.Module):
    layer = dict(model.named_children()).get('layer4')[-1].conv3
    return layer


def get_layer(model_name:str,
              model:nn.Module):
    
    if model_name == 'resnet50':
        layer = get_resnet50_layer(model)
    # add models...
    else:
        raise ValueError(f'{model_name} is not implemented')
    
    print('successfully extracted the layers')
    return layer



if __name__ == '__main__':
    print(get_layer('resnet50', models.resnet50(pretrained=True)))