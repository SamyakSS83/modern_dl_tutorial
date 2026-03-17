"""Training scaffold for A03 with CE/Focal options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from focal_loss import FocalLoss
from se_block import SEResNet18


# EuroSAT normalization (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Get augmentation pipeline."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class EuroSATDataset:
    """EuroSAT dataset wrapper with splits from splits.json."""

    def __init__(self, data_dir: str, split: str = "train", image_size: int = 224):
        self.base = datasets.EuroSAT(root=str(Path(data_dir)), download=False,
                                     transform=get_transforms(split, image_size))
        splits_file = Path(data_dir) / "splits.json"
        with open(splits_file, "r") as f:
            split_data = json.load(f)
        self.indices = split_data[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A03 model")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def eval_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute one-vs-rest ROC-AUC."""
    model.eval()
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs_all.append(torch.softmax(logits, dim=1).cpu())
            labels_all.append(y)
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return float(roc_auc_score(labels, probs, multi_class="ovr"))


def main(args: argparse.Namespace) -> None:
    """Run training and save best model by validation ROC-AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load EuroSAT train and val splits
    train_ds = EuroSATDataset(args.data_dir, split="train", image_size=args.image_size)
    val_ds = EuroSATDataset(args.data_dir, split="val", image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SEResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss() if args.loss == "ce" else FocalLoss(
        gamma=args.gamma
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)
        val_auc = eval_auc(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metric": val_auc,
                },
                Path(args.save_dir) / "best.pt",
            )
        
        # Always save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metric": val_auc,
            },
            Path(args.save_dir) / "last.pt",
        )


if __name__ == "__main__":
    main(parse_args())
