"""Verify the training environment before running anything else.

Run this first on the cluster: python -m src.utils.env_check
Fails loudly (non-zero exit) if CUDA is unavailable, so a misconfigured
environment doesn't silently fall back to CPU training.
"""

import sys

import torch
from monai.config import print_config


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print_config()

    if not torch.cuda.is_available():
        print("\nCUDA is NOT available -- training would silently run on CPU.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"\nCUDA available: {device_count} device(s)")
    for i in range(device_count):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
