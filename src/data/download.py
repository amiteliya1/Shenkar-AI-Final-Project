"""Download the MSD Task09_Spleen dataset.

Reuses MONAI's DecathlonDataset purely for its download/extract/checksum-verify
side effect. The transform is intentionally left empty and caching disabled: we
don't want any real data loading here, just the files on disk. The dataset
object itself is discarded -- the actual train/val split and transforms are a
separate, deliberate decision made in src/data/dataset.py (Day 2).
"""

import argparse

from monai.apps import DecathlonDataset

TASK = "Task09_Spleen"


def main(data_dir: str) -> None:
    DecathlonDataset(
        root_dir=data_dir,
        task=TASK,
        section="training",
        download=True,
        transform=(),
        cache_num=0,
        num_workers=4,
    )
    print(f"{TASK} is ready under {data_dir}/{TASK}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Download {TASK} from the Medical Segmentation Decathlon")
    parser.add_argument("--data-dir", default="data", help="Directory to download and extract the dataset into")
    args = parser.parse_args()
    main(args.data_dir)
