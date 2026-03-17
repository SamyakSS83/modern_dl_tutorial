"""Download EuroSAT and create deterministic splits for A04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import EuroSAT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download EuroSAT and split")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download EuroSAT and write split index JSON."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = EuroSAT(root=str(data_dir), download=True)
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)

    split = {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train : n_train + n_val].tolist(),
        "test": idx[n_train + n_val :].tolist(),
        "classes": ds.classes,
    }
    with (data_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Saved splits to {data_dir / 'splits.json'}")


if __name__ == "__main__":
    main(parse_args())
