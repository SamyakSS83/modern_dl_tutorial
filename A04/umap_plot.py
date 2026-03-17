"""UMAP plot for A04 encoder embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from umap import UMAP


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class EuroSATSplit(torch.utils.data.Dataset):
    """EuroSAT split wrapper based on splits.json."""

    def __init__(self, data_dir: str, split: str, image_size: int):
        self.base = datasets.EuroSAT(
            root=data_dir,
            download=False,
            transform=eval_transform(image_size),
        )
        with Path(data_dir, "splits.json").open("r", encoding="utf-8") as f:
            split_data = json.load(f)
        self.indices = split_data[split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="UMAP embedding visualization")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--simclr_encoder_ckpt", type=str, required=True)
    p.add_argument("--supervised_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def extract_embeddings(model, loader, device, max_samples: int):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
            if sum(len(b) for b in labels) >= max_samples:
                break
    x = np.concatenate(feats, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]
    return x, y


def save_umap(x: np.ndarray, y: np.ndarray, title: str, save_path: Path) -> None:
    emb = UMAP(n_components=2, random_state=42).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, s=8, cmap="tab10")
    ax.set_title(title)
    fig.colorbar(sc)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    """Extract features, fit UMAP, and save scatter plot."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = EuroSATSplit(
        data_dir=args.data_dir,
        split="test",
        image_size=args.image_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    simclr_encoder = models.resnet18(weights=None)
    simclr_encoder.fc = torch.nn.Identity()
    simclr_ckpt = torch.load(args.simclr_encoder_ckpt, map_location=device)
    simclr_encoder.load_state_dict(simclr_ckpt["model_state"])
    simclr_encoder.to(device).eval()

    x_simclr, y = extract_embeddings(
        simclr_encoder, loader, device, max_samples=args.max_samples
    )
    save_umap(
        x_simclr,
        y,
        "UMAP of SimCLR Encoder Features",
        save_dir / "umap_simclr.png",
    )

    supervised_model = models.resnet18(weights=None)
    supervised_model.fc = torch.nn.Linear(supervised_model.fc.in_features, 10)
    sup_ckpt = torch.load(args.supervised_ckpt, map_location=device)
    supervised_model.load_state_dict(sup_ckpt["model_state"])
    supervised_model.fc = torch.nn.Identity()
    supervised_model.to(device).eval()

    x_sup, y_sup = extract_embeddings(
        supervised_model, loader, device, max_samples=args.max_samples
    )
    save_umap(
        x_sup,
        y_sup,
        "UMAP of Supervised (10%) Features",
        save_dir / "umap_supervised.png",
    )


if __name__ == "__main__":
    main(parse_args())
