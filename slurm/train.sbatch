#!/bin/bash
# Trains any model supported by src/train.py's TrainConfig.model (currently
# "unet" or "swin_unetr") on the gpu partition, using the project's .venv.
# Config-driven and model-agnostic -- renamed from train_baseline.sbatch since
# it's no longer specific to the baseline.
#
# Usage:
#   sbatch slurm/train.sbatch [path-to-config.yaml]
# Defaults to configs/baseline_unet.yaml if no config is given, e.g.:
#   sbatch slurm/train.sbatch configs/swin_unetr_smoketest.yaml
#
# --gres=gpu:L4:1 is confirmed working (Slurm job 316, baseline U-Net run).
#
# --cpus-per-task=4 matches num_workers=4 in the config files -- keep these in
# sync if you change one.
# --mem and --time are conservative placeholders, not measured values -- adjust
# after seeing real usage from the smoke test (`sacct`, see README/chat notes).

#SBATCH --job-name=segmentation_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=SlurmLogs/%x_%j.out
#SBATCH --error=SlurmLogs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

CONFIG="${1:-configs/baseline_unet.yaml}"
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Running: python -m src.train --config $CONFIG"
python -m src.train --config "$CONFIG"
