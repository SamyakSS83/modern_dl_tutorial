"""Linear probe training for A04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class EuroSATSplit(torch.utils.data.Dataset):
    """EuroSAT split wrapper based on precomputed splits.json."""

    def __init__(self, data_dir: str, split: str, image_size: int, train_aug: bool):
        self.base = datasets.EuroSAT(
            root=data_dir,
            download=False,
            transform=get_transforms(image_size=image_size, train=train_aug),
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
    p = argparse.ArgumentParser(description="Linear probe on frozen encoder")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--sup_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def subset_with_fraction(dataset, frac: float, seed: int):
    n = max(1, int(frac * len(dataset)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    return Subset(dataset, perm[:n])


def train_fraction(
    frac: float,
    encoder: nn.Module,
    train_ds,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Train probe on a fraction of labels and return test accuracy."""
    subset = subset_with_fraction(train_ds, frac, args.seed)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    head = nn.Linear(512, 10).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    encoder.eval()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = encoder(x)
            logits = head(feats)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = head(encoder(x))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_supervised_fraction(
    frac: float,
    train_ds,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Train a supervised baseline on the same label fraction."""
    subset = subset_with_fraction(train_ds, frac, args.seed)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for _ in range(args.sup_epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return correct / max(total, 1), state


def main(args: argparse.Namespace) -> None:
    """Evaluate linear probing across label fractions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_ds = EuroSATSplit(
        args.data_dir, split="train", image_size=args.image_size, train_aug=True
    )
    test_ds = EuroSATSplit(
        args.data_dir, split="test", image_size=args.image_size, train_aug=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.to(device).eval()

    fracs = [0.01, 0.05, 0.1, 1.0]
    simclr_accs = [
        train_fraction(f, encoder, train_ds, test_loader, args, device)
        for f in fracs
    ]
    sup_accs: list[float] = []
    for f in fracs:
        sup_acc, sup_state = train_supervised_fraction(
            f, train_ds, test_loader, args, device
        )
        sup_accs.append(sup_acc)
        if abs(f - 0.1) < 1e-8:
            torch.save(
                {
                    "model_state": sup_state,
                    "fraction": f,
                    "image_size": args.image_size,
                },
                Path(args.save_dir) / "supervised_10pct.pt",
            )
        print(
            f"Fraction {f:.2f} | SimCLR probe acc: {simclr_accs[len(sup_accs)-1]:.4f} "
            f"| Supervised acc: {sup_acc:.4f}"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, simclr_accs, marker="o", label="SimCLR + linear probe")
    ax.plot(fracs, sup_accs, marker="s", label="Supervised (same labels)")
    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Label Efficiency: SimCLR vs Supervised")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "label_efficiency_curve.png")


if __name__ == "__main__":
    main(parse_args())
