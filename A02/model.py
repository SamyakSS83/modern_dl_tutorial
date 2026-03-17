"""Model skeleton for A02 fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class FineTuner(nn.Module):
    """ResNet fine-tuning model for EuroSAT.

    Input shape:
        x: (B, 3, H, W)
    Output shape:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10, freeze: bool = True) -> None:
        """Initialize backbone and classifier head.

        Learner task:
            Configure freezing behavior and replace head as requested.
        """
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model backbone."""
        return self.backbone(x)

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
