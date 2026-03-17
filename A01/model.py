"""Model skeleton for A01."""

from __future__ import annotations

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """Simple CNN for MNIST classification.

    Input shape:
        x: (B, 1, 28, 28)
    Output shape:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize model layers.

        Args:
            num_classes: Number of target classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of images.

        Args:
            x: Input image tensor of shape (B, 1, 28, 28).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)
