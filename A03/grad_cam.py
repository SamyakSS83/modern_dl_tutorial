"""Grad-CAM utilities for A03."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM helper for a target convolutional layer."""

class GradCAM:
    """Grad-CAM helper for a target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize and register hooks.

        Learner task:
            Register forward/backward hooks to cache activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Cache activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Cache gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap for a single image."""
        # Forward pass
        self.model.eval()
        x.requires_grad = True
        logits = self.model(x)
        
        # If no class specified, use predicted class
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Backward pass for the target class
        target_score = logits[0, class_idx]
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]      # (C, H, W)
        
        # Compute importance weights (average pooling of gradients)
        weights = gradients.mean(dim=[1, 2])  # (C,)
        
        # Weighted sum of activations
        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)
        
        # ReLU to keep only positive contributions
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image in uint8 RGB format."""
    h, w = image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = np.clip(0.6 * image.astype(np.float32) + 0.4 * color, 0, 255)
    return out.astype(np.uint8)
