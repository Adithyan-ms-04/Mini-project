import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAMPlusPlus:
    """
    Implements Grad-CAM++ for visualizing model attention.
    Specifically targets the last convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, left_img, right_img, class_idx=0):
        self.model.zero_grad()
        output = self.model(left_img, right_img)
        
        # Binary case
        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ calculation
        b, c, h, w = gradients.shape
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).sum((2,3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        
        weights = (alphas * torch.clamp(gradients, min=0)).sum((2,3), keepdim=True)
        heatmap = (weights * activations).sum(1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        heatmap = cv2.resize(heatmap, (300, 300))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()
