"""Utility functions for A02."""

from __future__ import annotations

import random

import numpy as np
import torch
from torchvision import transforms


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(split: str, image_size: int) -> transforms.Compose:
    """Build split-specific transforms for EuroSAT images."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def compute_metrics(
    preds: list[int], labels: list[int], class_names: list[str]
) -> dict[str, float]:
    """Compute accuracy and macro F1 metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    for class_idx, class_name in enumerate(class_names):
        mask = [i for i, y in enumerate(labels) if y == class_idx]
        if not mask:
            metrics[f"acc_{class_name}"] = 0.0
            continue
        correct = sum(int(preds[i] == labels[i]) for i in mask)
        metrics[f"acc_{class_name}"] = correct / len(mask)
    return metrics
