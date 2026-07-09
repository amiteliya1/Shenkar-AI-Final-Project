#!/bin/bash
# Trains a segmentation model on the gpu partition using the project's .venv.
#
# Usage:
#   sbatch slurm/train_baseline.sbatch [path-to-config.yaml]
# Defaults to configs/baseline_unet.yaml if no config is given, e.g.:
#   sbatch slurm/train_baseline.sbatch configs/baseline_unet_smoketest.yaml
#
# NOTE: --gres=gpu:L4:1 assumes the cluster's GRES name for an L4 GPU is
# literally "L4" -- verify with `sinfo -o "%P %G"` or `scontrol show partition gpu`
# and adjust if the cluster uses a different string (e.g. "l4", "nvidia_l4",
# or plain --gres=gpu:1 if the gpu partition is L4-only already).
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
