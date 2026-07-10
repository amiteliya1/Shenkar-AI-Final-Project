#!/bin/bash
# Generates Swin UNETR attention/importance explainability figures for the
# strong/average/weak validation cases (auto-selected from eval_results.json).
#
# Usage:
#   sbatch slurm/explain.sbatch [config.yaml] [checkpoint.pt] [output-dir]
# Defaults match the completed swin_unetr_v1 run, e.g. just:
#   sbatch slurm/explain.sbatch
#
# GPU is needed here (not just for training): the Grad-CAM computation runs
# a real forward+backward pass of Swin UNETR, same as a training step.
#
# --time=00:30:00: this only processes 3 cases (one sliding-window forward
# pass + one small-crop forward+backward each), far cheaper than training.

#SBATCH --job-name=segmentation_explain
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

CONFIG="${1:-configs/swin_unetr_base.yaml}"
CHECKPOINT="${2:-outputs/swin_unetr_v1/best_model.pt}"
OUTPUT_DIR="${3:-}"

echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ -n "$OUTPUT_DIR" ]]; then
  echo "Running: python -m src.explainability.attention_map --config $CONFIG --checkpoint $CHECKPOINT --output-dir $OUTPUT_DIR"
  python -m src.explainability.attention_map --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR"
else
  echo "Running: python -m src.explainability.attention_map --config $CONFIG --checkpoint $CHECKPOINT"
  python -m src.explainability.attention_map --config "$CONFIG" --checkpoint "$CHECKPOINT"
fi
