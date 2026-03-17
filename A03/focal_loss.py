"""Focal loss skeleton for A03."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss module for multi-class classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        """Initialize loss hyperparameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

class FocalLoss(nn.Module):
    """Focal loss module for multi-class classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        """Initialize loss hyperparameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, C)
            targets: (B,)
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get probabilities of the true class
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Combine: alpha * (1 - p_t)^gamma * ce_loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
