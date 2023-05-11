import torch
import cv2
import numpy as np

def normalize(tensor, mean, std):
    if not tensor.ndim == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


def visualize_cam(mask, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.cpu().squeeze()), cv2.COLORMAP_JET) # H, W, C
    heatmap = torch.from_numpy(heatmap.astype(np.float32)).permute(2, 0, 1).div(255) # C, H, W
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) # BGR to RGB
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result