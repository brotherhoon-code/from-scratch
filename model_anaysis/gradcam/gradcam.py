# ref: https://github.com/1Konny/gradcam_plus_plus-pytorch

import torch
import torch.nn.functional as F
import torch.nn as nn
from layer_extractor import get_layer


class GradCAM(object):
    def __init__(self, model_name: str, model: nn.Module):
        model_name = model_name
        self.model = model

        self.hooked_gradients = None
        self.hooked_activations = None

        def backward_hook(module, grad_input, grad_output):
            self.hooked_gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.hooked_activations = output

        # get from model_type, layer_name, model_arch
        target_layer: nn.Module = get_layer(model_name, model)  # type hinting

        # target_layer.register
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, img: torch.Tensor):
        """
        이미지를 받아 saliency_map과 logit을 리턴하는 함수

        Args:
            img (torch.Tensor): 이미지 텐서

        Returns:
            sliency_map
            logit
        """
        B, C, H, W = img.size()
        logit: torch.Tensor = self.model(img)
        score = logit[:, logit.max(1)[-1]].squeeze()  # get max class score
        self.model.zero_grad()
        score.backward()

        # get hooked grad, activation
        gradients: torch.Tensor = self.hooked_gradients
        activations: torch.Tensor = self.hooked_activations

        B, K, U, V = gradients.size()

        # calc mean of channel-wise gradient
        alpha = gradients.view(B, K, -1).mean(2)
        weights = alpha.view(B, K, 1, 1)  # reshape B, C, 1, 1

        # calc saliency map by alpha & weight
        saliency_map = (weights * activations).sum(1, keepdim=True)  # B, 1, h, w
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(
            saliency_map, size=(H, W), mode="bilinear", align_corners=False
        )  # B, 1, H, W
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            (saliency_map - saliency_map_min)
            .div(saliency_map_max - saliency_map_min)
            .data
        )
        return saliency_map, logit

    def __call__(self, img: torch.Tensor):
        return self.forward(img)
