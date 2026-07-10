#!/bin/bash
# Evaluates a trained checkpoint (Dice + HD95) on the shared validation split.
#
# Usage:
#   sbatch slurm/evaluate.sbatch <config.yaml> <checkpoint.pt> [output.json]
# e.g.:
#   sbatch slurm/evaluate.sbatch configs/baseline_unet.yaml outputs/baseline_unet_v1/best_model.pt
#   sbatch slurm/evaluate.sbatch configs/swin_unetr_base.yaml outputs/swin_unetr_v1/best_model.pt
#
# --time=00:30:00: evaluation is just forward passes (no training) over ~8
# validation volumes, far cheaper than a training run -- 30 minutes is a
# generous margin, and matches the effective ceiling already observed on this
# cluster's gpu partition regardless (see slurm/train.sbatch's note on job 322).

#SBATCH --job-name=segmentation_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=SlurmLogs/%x_%j.out
#SBATCH --error=SlurmLogs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

CONFIG="$1"
CHECKPOINT="$2"
OUTPUT="${3:-}"

echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ -n "$OUTPUT" ]]; then
  echo "Running: python -m src.evaluate --config $CONFIG --checkpoint $CHECKPOINT --output $OUTPUT"
  python -m src.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT" --output "$OUTPUT"
else
  echo "Running: python -m src.evaluate --config $CONFIG --checkpoint $CHECKPOINT"
  python -m src.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT"
fi
