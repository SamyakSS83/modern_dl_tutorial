"""Sanity check script for NT-Xent implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from simclr import NTXentLoss


if __name__ == "__main__":
    z1 = torch.randn(8, 128)
    z2 = z1 + 0.01 * torch.randn_like(z1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    loss_fn = NTXentLoss(temperature=0.2)
    loss_same = loss_fn(z1, z1)
    loss_near = loss_fn(z1, z2)
    loss_rand = loss_fn(z1, F.normalize(torch.randn_like(z1), dim=1))
    print(f"Loss (identical views): {loss_same.item():.4f}")
    print(f"Loss (near views):      {loss_near.item():.4f}")
    print(f"Loss (random views):    {loss_rand.item():.4f}")
    print(f"Expected random ~ log(2N-1): {math.log(2 * z1.size(0) - 1):.4f}")
