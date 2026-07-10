#!/bin/bash
# Evaluates a trained checkpoint (Dice + HD95) on the shared validation split.
#
# Usage:
#   sbatch slurm/evaluate.sbatch <config.yaml> <checkpoint.pt> [output.json] [postprocess]
# e.g.:
#   sbatch slurm/evaluate.sbatch configs/baseline_unet.yaml outputs/baseline_unet_v1/best_model.pt
#   sbatch slurm/evaluate.sbatch configs/swin_unetr_base.yaml outputs/swin_unetr_v1/best_model.pt
#   sbatch slurm/evaluate.sbatch configs/swin_unetr_base.yaml outputs/swin_unetr_v1/best_model.pt "" postprocess
# The optional 4th arg, the literal word "postprocess", adds --postprocess
# (largest-connected-component cleanup -- see src/evaluate.py's docstring).
# Pass "" for the 3rd arg to use the default output path while still setting this.
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
POSTPROCESS="${4:-}"

ARGS=(--config "$CONFIG" --checkpoint "$CHECKPOINT")
if [[ -n "$OUTPUT" ]]; then
  ARGS+=(--output "$OUTPUT")
fi
if [[ "$POSTPROCESS" == "postprocess" ]]; then
  ARGS+=(--postprocess)
fi

echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Running: python -m src.evaluate ${ARGS[*]}"
python -m src.evaluate "${ARGS[@]}"
