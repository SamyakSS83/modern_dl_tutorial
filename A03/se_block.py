"""SE module and SE-ResNet skeletons for A03."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Input/output shape:
        x: (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SE block components."""
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to feature map."""
        b, c, h, w = x.shape
        # Squeeze: global average pooling
        z = self.squeeze(x)  # (B, C, 1, 1)
        z = z.view(b, c)  # (B, C)
        # Excitation: learned channel weights
        s = self.excitation(z)  # (B, C)
        s = s.view(b, c, 1, 1)  # (B, C, 1, 1)
        # Scale: channel-wise multiplication
        return x * s


class SEResNet18(nn.Module):
    """ResNet-18 with SE blocks inserted after major stages."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize backbone and placeholders for SE insertion."""
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        # Create four SE blocks matching layer1..layer4 channel widths.
        self.se1 = SEBlock(64, reduction=16)
        self.se2 = SEBlock(128, reduction=16)
        self.se3 = SEBlock(256, reduction=16)
        self.se4 = SEBlock(512, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with SE insertion points."""
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.se1(x)
        x = self.base.layer2(x)
        x = self.se2(x)
        x = self.base.layer3(x)
        x = self.se3(x)
        x = self.base.layer4(x)
        x = self.se4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        return self.base.fc(x)
