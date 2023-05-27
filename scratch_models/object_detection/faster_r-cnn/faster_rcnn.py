import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import nms
import numpy as np
from utils import totensor, loc2bbox, tonumpy
from backbone import decomp_vgg16
from rpn import RegionProposalNetwork
from head import VGG16RoIHead


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args,**kwargs)
    return new_f


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                 loc_norm_mean = (0. ,0. ,0. ,0.),
                 loc_norm_std = (0.1, 0.1, 0.2, 0.2),
                 learning_rate = 1e-3, 
                 lr_decay = 0.1,
                 weight_decay = 0.0005):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        
        self.loc_norm_mean = loc_norm_mean
        self.loc_norm_std = loc_norm_std
        self.use_preset()
    
    @property
    def n_classes(self):
        return self.head.n_classes
    
    def forward(self, x:torch.Tensor, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices
    
    def use_preset(self):
        self.nms_thresh = 0.3
        self.score_thresh = 0.05
    
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        
        for l in range(1, self.n_classes):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_classes, 4))[:,1,:]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l-1)*np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
            
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
        
    @nograd
    def predict(self, imgs, sizes=None):
        self.eval()
        prepared_imgs = imgs
        
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, sclae=scale)
            
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois)/scale
            
            mean = torch.Tensor(self.loc_norm_mean).cuda().repeat(self.n_classes)[None]
            std = torch.Tensor(self.loc_norm_std).cuda().repeat(self.n_classes)[None]
            
            roi_cls_loc:torch.Tensor = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_classes, 4)
            roi:torch.Tensor = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1,4)),
                                tonumpy(roi_cls_loc).reshape((-1,4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_classes*4)
            # clip bboxes
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            
            prob = (F.softmax(totensor(roi_score), dim=1))
            
            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        
        self.use_preset()
        self.train()
        return bboxes, labels, scores
        
    def get_optimizer(self):
        lr = self.learning_rate
        params = list()
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value], 'lr':lr*2, 'weight_decay':0}]
                else:
                    params += [{'params':[value], 'lr':lr, 'weight_decay':self.weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer
        
    def scale_lr(self, decay=0.1):
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] *= decay
        return self.optimizer


class FasterRCNNVGG16(FasterRCNN):
    
    feat_stride = 16
    
    def __init__(self,
                 n_fg_classes = 10,
                 ratios=[0.5,1,2],
                 anchor_scales=[8,16,32]):
        extractor, classifier = decomp_vgg16()
        
        rpn = RegionProposalNetwork(512,512,
                                    ratios=ratios,
                                    anchor_scales=anchor_scales,
                                    feat_stride=self.feat_stride)
        
        head = VGG16RoIHead(n_classes=n_fg_classes+1,
                            roi_size=7,
                            spatial_scale=(1./self.feat_stride),
                            classifier=classifier)
        
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)
    