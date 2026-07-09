"""Sanity-check the data pipeline before any training.

Loads one validation-pipeline volume (no random cropping, so the whole scan is
visible) and saves a middle-axial-slice figure with the spleen mask overlaid.
Run this after download.py and before train.py -- if the mask doesn't align
with a plausible spleen-shaped region, something in transforms.py is wrong
(orientation, spacing, or intensity windowing) and training would silently
learn from broken data.

Usage: python -m src.data.visualize_sample --data-dir data --index 0
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import get_datalists
from src.data.transforms import get_val_transforms


def main(data_dir: str, index: int, out_path: str) -> None:
    _, val_files = get_datalists(data_dir)
    sample = get_val_transforms()(val_files[index])

    image = sample["image"][0].numpy()  # (H, W, D), channel dim already stripped
    label = sample["label"][0].numpy()

    mid_slice = image.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[:, :, mid_slice], cmap="gray")
    axes[0].set_title("CT slice (windowed)")
    axes[0].axis("off")

    axes[1].imshow(image[:, :, mid_slice], cmap="gray")
    axes[1].imshow(np.ma.masked_where(label[:, :, mid_slice] == 0, label[:, :, mid_slice]), cmap="autumn", alpha=0.5)
    axes[1].set_title("Spleen mask overlay")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved sanity-check figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize one preprocessed volume+mask pair")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", type=int, default=0, help="Validation-set index to visualize")
    parser.add_argument("--out", default="outputs/sanity_check.png")
    args = parser.parse_args()
    main(args.data_dir, args.index, args.out)
