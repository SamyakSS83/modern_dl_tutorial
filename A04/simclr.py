"""Core SimCLR modules for A04."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder features to contrastive space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize projection layers.

        Learner task:
            Build a 2-layer MLP projection head.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output features."""
        return self.net(x)


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss."""

    def __init__(self, temperature: float = 0.2) -> None:
        """Initialize NT-Xent with temperature scaling."""
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss for positive pairs.

        Args:
            z1: (B, D) projected features for view 1.
            z2: (B, D) projected features for view 2.
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.t()) / self.temperature
        idx = torch.arange(2 * batch_size, device=z.device)
        pos_idx = (idx + batch_size) % (2 * batch_size)
        positives = sim[idx, pos_idx].unsqueeze(1)

        neg_mask = torch.ones_like(sim, dtype=torch.bool)
        neg_mask[idx, idx] = False
        neg_mask[idx, pos_idx] = False
        negatives = sim[neg_mask].view(2 * batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)


class SimCLR(nn.Module):
    """Wrapper around encoder and projection head."""

    def __init__(self, encoder: nn.Module, projector: ProjectionHead) -> None:
        """Store encoder and projector."""
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project a batch of images."""
        h = self.encoder(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        return self.projector(h)
