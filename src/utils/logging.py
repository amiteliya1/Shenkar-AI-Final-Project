"""Lightweight local experiment logging: CSV + matplotlib, no external service.

No W&B/MLflow dependency -- keeps every run reproducible offline and avoids
depending on outbound network access from the university cluster.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


class RunLogger:
    """Accumulates per-epoch metrics for one run and writes them to disk."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self._rows: list[dict] = []

    def log(self, epoch: int, train_loss: float, val_dice: Optional[float] = None) -> None:
        self._rows.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
        self._write_csv()

    def _write_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_dice"])
            writer.writeheader()
            writer.writerows(self._rows)

    def plot(self, out_path: Optional[str] = None) -> None:
        """Save a learning-curve figure: train loss and validation Dice vs. epoch."""
        epochs = [r["epoch"] for r in self._rows]
        train_loss = [r["train_loss"] for r in self._rows]
        val_epochs = [r["epoch"] for r in self._rows if r["val_dice"] is not None]
        val_dice = [r["val_dice"] for r in self._rows if r["val_dice"] is not None]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, train_loss, color="tab:red", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(val_epochs, val_dice, color="tab:blue", marker="o", label="val Dice")
        ax2.set_ylabel("val Dice", color="tab:blue")

        fig.tight_layout()
        out_path = out_path or str(self.run_dir / "learning_curve.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
