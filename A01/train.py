"""Training scaffold for A01."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MyCNN
from utils import get_device, plot_curves, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A01 CNN on MNIST")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model on validation split."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main(args: argparse.Namespace) -> None:
    """Run train/validation loop skeleton."""
    set_seed(args.seed)
    device = get_device()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=False,
    )
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    train_ds, val_ds = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
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

    model = MyCNN(num_classes=10).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        _, val_acc = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metric": val_acc,
        }
        torch.save(ckpt, Path(args.save_dir) / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, Path(args.save_dir) / "best.pt")

    plot_curves(train_losses, val_accs, str(Path(args.save_dir) / "curves.png"))


if __name__ == "__main__":
    main(parse_args())
