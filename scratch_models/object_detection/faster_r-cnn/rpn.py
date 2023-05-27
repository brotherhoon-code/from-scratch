import numpy as np
import torch.nn as nn
import torch
import six
from utils import bbox2loc, loc2bbox, normal_init, _eu
from torchvision.ops import nms
from torch.nn import functional as F

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8,16,32]):
    
    py = base_size / 2. # center y
    px = base_size / 2. # center x
    
    anchor_base = np.zeros((len(ratios)*len(anchor_scales), 4), dtype=np.float32)
    
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1./ratios[i])
            
            index = i * len(anchor_scales) + j
            
            anchor_base[index, 0] = py - h/2.
            anchor_base[index, 1] = px - w/2.
            anchor_base[index, 2] = py + h/2.
            anchor_base[index, 3] = py + w/2.
    
    return anchor_base

class ProposalCreator:
    def __init__(self, parent_model:nn.Module,
                 nms_thresh=0.7, # nms threshold
                 n_train_pre_nms=12000, # train시 nms 전 roi 개수
                 n_train_post_nms=2000, # train시 nms 후 roi 개수
                 n_test_pre_nms=6000, # test시 nms 전 roi 개수
                 n_test_post_nms=300, # test시 nms 후 roi 개수
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
        
    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training: # train
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else: # test
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
            
        roi = loc2bbox(anchor, loc)
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])
        
        min_size = self.min_size*scale
        hs = roi[:,2] - roi[:,0]
        ws = roi[:,3] - roi[:,1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        order = score.ravel().argsort()[::-1]
        
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        
        # nms 적용
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)

        if n_post_nms>0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        
        return roi

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5,1,2],
                 anchor_scales=[8,16,32], feat_stride=16, proposal_creator_params=dict(),):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=3, stride=1, padding=1)
        self.score = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor*2,
                               kernel_size=1, stride=1, padding=0)
        self.loc = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor*4,
                             kernel_size=1, stride=1, padding=0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        
        
    def forward(self, x:torch.Tensor, img_size, scale=1.):
        n, _, hh, ww = x.shape
        
        anchor = self._enumerate_shifted_anchor(np.array(self.anchor_base),
                                                self.feat_stride,
                                                hh,ww)
        n_anchor = anchor.shape[0] // (hh*ww)
        middle = F.relu(self.conv1(x))
        
        # predict bboxes offsets
        rpn_locs = self.loc(middle)
        rpn_locs = rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        
        # predicted scores for anchor
        rpn_scores = self.score(middle)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        
        rpn_softmax_scores = F.softmax(rpn_scores.view(n,hh,ww,n_anchor,2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n,-1)
        
        # scores for foreground
        rpn_scores = rpn_scores.view(n,-1,2)
        
        # proposal
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),
                                      rpn_fg_scores[i].cpu().data.numpy(),
                                      img_size,
                                      sclae=scale)
            batch_index = i*np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    
    
    def _enumerate_shifted_anchor(anchor_base:np.ndarray, feat_stride, height, width):
        shift_y = np.arange(0, height*feat_stride, feat_stride)
        shift_x = np.arange(0, width*feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_x.ravel()), axis=1)
        A = anchor_base.shape[0]
        K = shift.shape[0]
        anchor = anchor_base.reshape((1, A, 4)) + \
            shift.reshape((1,K,4)).transpose((1,0,2))
        anchor = anchor.reshape((K*A,4)).astype(np.float32)
        return anchor