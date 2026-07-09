"""Experiment configuration loading.

Each YAML file under configs/ fully specifies one run (model choice + training
hyperparameters). Preprocessing constants (HU window, spacing, patch size) are
NOT part of this config -- they live in src/data/transforms.py since both
models must share the exact same data pipeline for the baseline-vs-Swin-UNETR
comparison to be meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    run_name: str
    model: str  # "unet" (Day 3) or "swin_unetr" (Day 5)
    data_dir: str = "data"
    max_epochs: int = 100
    val_interval: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5  # L2 regularization (Ch.3)
    dropout: float = 0.2  # Ch.3 regularization
    early_stopping_patience: int = 20  # in epochs, Ch.3 regularization
    num_workers: int = 4
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
