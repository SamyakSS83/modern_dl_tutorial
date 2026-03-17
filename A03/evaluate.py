"""Evaluation and Grad-CAM export for A03."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from grad_cam import GradCAM, overlay_heatmap
from se_block import SEResNet18


# EuroSAT normalization (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Get evaluation augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class EuroSATDataset:
    """EuroSAT dataset wrapper with splits from splits.json."""

    def __init__(self, data_dir: str, split: str = "test", image_size: int = 224):
        self.base = datasets.EuroSAT(root=str(Path(data_dir)), download=False,
                                     transform=get_transforms(image_size))
        self.base_raw = datasets.EuroSAT(root=str(Path(data_dir)), download=False,
                                         transform=None)
        splits_file = Path(data_dir) / "splits.json"
        with open(splits_file, "r") as f:
            split_data = json.load(f)
        self.indices = split_data[split]
        self.transform_unnorm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
    
    def get_unnormalized(self, idx):
        """Get unnormalized image for visualization."""
        actual_idx = self.indices[idx]
        img, label = self.base_raw[actual_idx]
        return self.transform_unnorm(img), label


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate A03 checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Compute metrics and save Grad-CAM grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load test dataset
    test_ds = EuroSATDataset(args.data_dir, split="test", image_size=args.image_size)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = SEResNet18(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Compute metrics
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            probs_all.append(probs)
            labels_all.append(y.numpy())
            preds_all.append(preds)

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = np.concatenate(preds_all)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, probs, multi_class='ovr')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # Generate Grad-CAM grid: one image per class
    cam = GradCAM(model, model.base.layer4)
    images = []
    for c in range(10):
        # Find first test image of class c that is correctly classified
        found = False
        for idx, actual_idx in enumerate(test_ds.indices):
            pred_label = preds[idx]
            true_label = labels[idx]
            if true_label == c and pred_label == c:
                # Get normalized image for model
                img_norm, _ = test_ds[idx]
                x = img_norm.unsqueeze(0).to(device)
                
                # Get unnormalized image for visualization
                img_unnorm, _ = test_ds.get_unnormalized(idx)
                img_display = (img_unnorm.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                
                # Generate Grad-CAM
                heat = cam.generate(x, class_idx=c)
                
                # Overlay and save
                images.append(overlay_heatmap(img_display, heat))
                found = True
                break
        
        if not found:
            # Fallback: use any image of class c
            for idx, actual_idx in enumerate(test_ds.indices):
                true_label = labels[idx]
                if true_label == c:
                    img_norm, _ = test_ds[idx]
                    x = img_norm.unsqueeze(0).to(device)
                    img_unnorm, _ = test_ds.get_unnormalized(idx)
                    img_display = (img_unnorm.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    heat = cam.generate(x, class_idx=c)
                    images.append(overlay_heatmap(img_display, heat))
                    break

    # Save 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"Class {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "gradcam_grid.png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Grad-CAM grid saved to {Path(args.save_dir) / 'gradcam_grid.png'}")


if __name__ == "__main__":
    main(parse_args())
