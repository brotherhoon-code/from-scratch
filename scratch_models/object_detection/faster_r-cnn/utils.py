import numpy as np
import torch.nn as nn
import torch

def loc2bbox(src_bbox:np.ndarray, loc:np.ndarray):
    """
    Decodes bboxes from bboxes offsets and scales

    Args:
        src_bbox: coordinates of bboxes
            (p_ymin, p_xmin, p_ymax, p_xmax)
        loc: an array with offsets and scales
            the shapes of the 'src_bbox' and 'loc' should be same.
            (t_y, t_x, t_h, t_h)
    Returns: Decoded bboxes coordinates
    """
    
    if src_bbox.shape[0] == 0: # 없을경우 차원만 맞춰서 리턴
        return np.zeros((0, 4), dtype=loc.dtype)
    
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_height = src_bbox[:,2] - src_bbox[:,0]
    src_width = src_bbox[:,3] - src_bbox[:,1]
    src_ctr_y = src_bbox[:,0] + 0.5*src_height
    src_ctr_x = src_bbox[:,1] + 0.5*src_width
    
    dy = loc[:,0::4]
    dx = loc[:,1::4]
    dh = loc[:,2::4]
    dw = loc[:,3::4]
    
    ctr_y = dy*src_height[:, np.newaxis] + src_ctr_y[:,np.newaxis]
    ctr_x = dx*src_width[:,np.newaxis] + src_ctr_x[:,np.newaxis]
    
    h = np.exp(dh)*src_height[:,np.newaxis]
    w = np.exp(dw)*src_width[:,np.newaxis]
    
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:,0::4] = ctr_y - 0.5*h
    dst_bbox[:,1::4] = ctr_x - 0.5*w
    dst_bbox[:,2::4] = ctr_y + 0.5*h
    dst_bbox[:,3::4] = ctr_x + 0.5*w
    
    return dst_bbox

def bbox2loc(src_bbox:np.array, dst_bbox:np.array) -> np.array:
    """
    Encodes the source and the destination bboxes to "loc".
    The offsets and scales t_y, t_x, t_h, t_w can be computed by the following formulas
    t_y = (g_y - p_y) / p_h
    t_x = (g_x - p_x) / p_w
    t_h = log(g_h/p_h)
    t_w = log(g_w/p_w)
    
    Args:
        src_bbox: (p_ymin, p_xmin, p_ymax, p_xmax)
        dst_bbox: (g_ymin, g_xmin, g_ymax, g_xmax)
    
    Returns:
        (t_y, t_x, t_h, t_w)
    """
    
    height = src_bbox[:,2]-src_bbox[:,0]
    width = src_bbox[:,3]-src_bbox[:,1]
    ctr_y = src_bbox[:,0] + 0.5*height
    ctr_x = src_bbox[:,1] + 0.5*width
    
    base_height = dst_bbox[:,2]-dst_bbox[:,0]
    base_width = dst_bbox[:,3]-dst_bbox[:1]
    base_ctr_y = dst_bbox[:,0]+0.5*base_height
    base_ctr_x = dst_bbox[:,1]+0.5*base_width
    
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height/height)
    dw = np.log(base_width/width)
    
    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def normal_init(m:nn.Module, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # 자기 자신을 변경하므로 normal_
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
        
def get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

def unmap(data:np.ndarray, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()