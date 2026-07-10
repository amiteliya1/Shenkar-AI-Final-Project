"""Evaluate a trained checkpoint (Dice + HD95) on the validation split.

    python -m src.evaluate --config configs/baseline_unet.yaml \\
        --checkpoint outputs/baseline_unet_v1/best_model.pt
    python -m src.evaluate --config configs/swin_unetr_base.yaml \\
        --checkpoint outputs/swin_unetr_v1/best_model.pt

Add --postprocess to additionally apply largest-connected-component
postprocessing before scoring (see src/utils/metrics.py's
make_largest_component_postprocess() and reports/experiment_log.md) --
writes to eval_results_postprocessed.json by default so it never overwrites
the raw result.

Reuses train.py's build_model() (so both "unet" and "swin_unetr" are
supported the same way training selects them) and dataset.py's
get_datalists() (the same fixed-seed 80/20 split, same
src/data/transforms.py preprocessing) -- this script does not construct any
new split or preprocessing of its own, so both models are evaluated on
exactly the same validation cases used during their training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from torch.cuda.amp import autocast

from src.data.dataset import get_datalists
from src.data.transforms import PATCH_SIZE, get_val_transforms
from src.train import build_model
from src.utils.config import TrainConfig
from src.utils.metrics import make_dice_metric, make_hd95_metric, make_largest_component_postprocess


def evaluate(config: TrainConfig, checkpoint_path: str, output_path: str, postprocess: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same split, same preprocessing as train.py's validation set -- no new split created.
    _, val_files = get_datalists(config.data_dir)
    val_ds = CacheDataset(
        data=val_files, transform=get_val_transforms(), cache_rate=1.0, num_workers=config.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=config.num_workers)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dice_metric = make_dice_metric()
    hd95_metric = make_hd95_metric()
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    keep_largest = make_largest_component_postprocess() if postprocess else None

    per_case = []
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        for case_files, batch in zip(val_files, val_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = sliding_window_inference(inputs=inputs, roi_size=PATCH_SIZE, sw_batch_size=4, predictor=model)
            outputs = [post_pred(x) for x in decollate_batch(outputs)]
            if keep_largest is not None:
                outputs = [keep_largest(x) for x in outputs]
            labels = [post_label(x) for x in decollate_batch(labels)]

            dice_value = dice_metric(y_pred=outputs, y=labels).item()
            hd95_value = hd95_metric(y_pred=outputs, y=labels).item()
            print(f"{Path(case_files['image']).stem}: dice={dice_value:.4f} hd95={hd95_value:.4f}")

            # Both metrics can be NaN/inf for a degenerate case (e.g. no predicted or
            # ground-truth foreground voxels at all) -- neither is valid JSON, so
            # store null instead of a bare inf/nan float, and exclude from the means below.
            per_case.append(
                {
                    "case_id": Path(case_files["image"]).stem,
                    "dice": dice_value if math.isfinite(dice_value) else None,
                    "hd95": hd95_value if math.isfinite(hd95_value) else None,
                }
            )

    valid_dice = [c["dice"] for c in per_case if c["dice"] is not None]
    valid_hd95 = [c["hd95"] for c in per_case if c["hd95"] is not None]

    summary = {
        "num_cases": len(per_case),
        "num_valid_dice": len(valid_dice),
        "mean_dice": (sum(valid_dice) / len(valid_dice)) if valid_dice else None,
        "std_dice": _std(valid_dice) if valid_dice else None,
        "num_valid_hd95": len(valid_hd95),
        "mean_hd95": (sum(valid_hd95) / len(valid_hd95)) if valid_hd95 else None,
        "std_hd95": _std(valid_hd95) if valid_hd95 else None,
    }

    results = {
        "run_name": config.run_name,
        "model": config.model,
        "checkpoint": str(checkpoint_path),
        "postprocess": "keep_largest_component" if postprocess else None,
        "per_case": per_case,
        "summary": summary,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    mean_dice_str = f"{summary['mean_dice']:.4f}" if summary["mean_dice"] is not None else "N/A"
    mean_hd95_str = f"{summary['mean_hd95']:.4f}" if summary["mean_hd95"] is not None else "N/A"
    print(f"\nmean_dice={mean_dice_str}  mean_hd95={mean_hd95_str}")
    print(f"Results written to {output_path}")


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint's Dice/HD95 on the validation split")
    parser.add_argument("--config", required=True, help="Path to the YAML config the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True, help="Path to a best_model.pt checkpoint")
    parser.add_argument("--output", default=None, help="Defaults to experiments/<run_name>/eval_results.json")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply largest-connected-component postprocessing to each prediction before "
        "scoring (candidate fix for stray false-positive blobs -- see reports/experiment_log.md). "
        "Defaults --output to eval_results_postprocessed.json instead of overwriting the raw result.",
    )
    args = parser.parse_args()

    train_config = TrainConfig.from_yaml(args.config)
    default_name = "eval_results_postprocessed.json" if args.postprocess else "eval_results.json"
    output = args.output or f"experiments/{train_config.run_name}/{default_name}"
    evaluate(train_config, args.checkpoint, output, postprocess=args.postprocess)
