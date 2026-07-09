"""Train/validation split and MONAI dataset construction for Task09_Spleen.

The split is a fixed-seed 80/20 random split over the 41 labeled training
volumes, so results are reproducible across machines and runs rather than
depending on which cases happen to get picked.
"""

import random
from pathlib import Path

from monai.data import CacheDataset, load_decathlon_datalist

from src.data.transforms import get_train_transforms, get_val_transforms

VAL_FRACTION = 0.2
SPLIT_SEED = 0


def get_datalists(data_dir: str) -> tuple[list, list]:
    """Return (train_files, val_files): lists of {"image": path, "label": path} dicts."""
    dataset_json = Path(data_dir) / "Task09_Spleen" / "dataset.json"
    datalist = load_decathlon_datalist(str(dataset_json), is_segmentation=True, data_list_key="training")

    shuffled = datalist.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    return shuffled[n_val:], shuffled[:n_val]


def get_datasets(data_dir: str, cache_rate: float = 1.0, num_workers: int = 4) -> tuple[CacheDataset, CacheDataset]:
    """Build the train/val MONAI CacheDatasets with their respective transform pipelines."""
    train_files, val_files = get_datalists(data_dir)
    train_ds = CacheDataset(
        data=train_files, transform=get_train_transforms(), cache_rate=cache_rate, num_workers=num_workers
    )
    val_ds = CacheDataset(
        data=val_files, transform=get_val_transforms(), cache_rate=cache_rate, num_workers=num_workers
    )
    return train_ds, val_ds
