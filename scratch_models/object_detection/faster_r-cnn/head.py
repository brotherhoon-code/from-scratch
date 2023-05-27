from torchvision.models import vgg16
from torchvision.ops import RoIPool # torchvision에서 제공하는 RoIPool
from torchvision.ops import nms
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import normal_init, totensor


class VGG16RoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_classes*4)
        self.score = nn.Linear(4096,n_classes)
        
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        
        self.n_classes = n_classes
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)
        
    def forward(self, x, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:,None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:,[0,2,1,4,3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        
        pool = self.roi(x, indices_and_rois) # roi pooling
        pool = pool.view(pool.size(0), -1) # flatten
        fc7 = self.classifier(pool) # fully connected
        roi_cls_locs = self.cls_loc(fc7) # regression
        roi_scores = self.score(fc7) # softmax
        
        return roi_cls_locs, roi_scores     
    
        