"""Augmentations for SimCLR (A04)."""

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def simclr_augmentation(image_size: int) -> transforms.Compose:
    """Build SimCLR augmentation pipeline."""
    blur_kernel = max(3, int(0.1 * image_size) | 1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=blur_kernel)], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TwoViewTransform:
    """Apply base transform twice to create a positive pair."""

    def __init__(self, base_transform: transforms.Compose) -> None:
        """Store base transform."""
        self.base_transform = base_transform

    def __call__(self, x):
        """Return two independently augmented views."""
        return self.base_transform(x), self.base_transform(x)
