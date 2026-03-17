"""EuroSAT data reference for A03.

Note: EuroSAT is downloaded in A02 and shared across assignments.
If you haven't run A02 yet, download EuroSAT with:
    python A02/download_data.py --data_dir data --seed 42 --train_frac 0.7 --val_frac 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import EuroSAT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download/verify EuroSAT for A03")
    p.add_argument("--data_dir", type=str, default="./data")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download EuroSAT if not already present."""
    data_path = Path(args.data_dir)
    if (data_path / "2750" / "AnnualCrop_1.jpg").exists():
        print(f"EuroSAT already downloaded to {args.data_dir}")
        return
    
    print(f"Downloading EuroSAT to {args.data_dir}")
    EuroSAT(root=str(data_path), download=True)
    print(f"EuroSAT downloaded successfully")


if __name__ == "__main__":
    main(parse_args())
