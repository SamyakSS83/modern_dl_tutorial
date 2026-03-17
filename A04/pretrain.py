"""SimCLR pretraining scaffold for A04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import EuroSAT

from augmentations import TwoViewTransform, simclr_augmentation
from simclr import NTXentLoss, ProjectionHead, SimCLR
from utils import get_device, set_seed


class EuroSATContrastiveDataset(torch.utils.data.Dataset):
    """EuroSAT dataset for contrastive pretraining with two-view transform."""

    def __init__(self, data_dir: str, transform, split_file: str = "splits.json"):
        self.base = EuroSAT(root=data_dir, download=False, transform=transform)
        split_path = Path(data_dir) / split_file
        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as f:
                split_data = json.load(f)
            self.indices = split_data["train"]
        else:
            self.indices = list(range(len(self.base)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Pretrain SimCLR")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run SimCLR pretraining loop skeleton."""
    set_seed(args.seed)
    device = get_device()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    aug = TwoViewTransform(simclr_augmentation(args.image_size))
    ds = EuroSATContrastiveDataset(data_dir=args.data_dir, transform=aug)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = models.resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    projector = ProjectionHead(in_dim=512, hidden_dim=256, out_dim=128)
    model = SimCLR(encoder=encoder, projector=projector).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        seen = 0
        for (x1, x2), _ in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            optim.zero_grad()
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2)
            loss.backward()
            optim.step()
            bs = x1.size(0)
            run_loss += loss.item() * bs
            seen += bs

        train_loss = run_loss / max(seen, 1)
        print(f"Epoch {epoch}/{args.epochs} | Contrastive Loss: {train_loss:.4f}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state": encoder.state_dict(),
            "optimizer_state": optim.state_dict(),
            "train_loss": train_loss,
            "image_size": args.image_size,
        },
        Path(args.save_dir) / "encoder.pt",
    )


if __name__ == "__main__":
    main(parse_args())
