"""Training script for A02 transfer learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EuroSATDataset
from model import FineTuner
from utils import compute_metrics, get_device, get_transforms, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A02 FineTuner")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_eval(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Compute validation metrics."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            p = out.argmax(dim=1)
            preds.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
    class_names = [str(i) for i in range(10)]
    return compute_metrics(preds, labels, class_names)


def main(args: argparse.Namespace) -> None:
    """Train and validate fine-tuning model."""
    set_seed(args.seed)
    device = get_device()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_ds = EuroSATDataset(
        args.data_dir,
        "train",
        get_transforms("train", args.image_size),
    )
    val_ds = EuroSATDataset(
        args.data_dir,
        "val",
        get_transforms("val", args.image_size),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FineTuner(num_classes=10, freeze=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    best_val = -1.0
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
        val_metrics = run_eval(model, val_loader, device)
        val_acc = val_metrics["accuracy"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metric": val_acc,
            }
            torch.save(ckpt, Path(args.save_dir) / "best.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metric": val_acc,
            },
            Path(args.save_dir) / "last.pt",
        )


if __name__ == "__main__":
    main(parse_args())
