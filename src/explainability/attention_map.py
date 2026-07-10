"""Attention/importance explainability for the trained Swin UNETR checkpoint.

    python -m src.explainability.attention_map \\
        --config configs/swin_unetr_base.yaml \\
        --checkpoint outputs/swin_unetr_v1/best_model.pt

METHOD (see "determine the safest way" in the project's Day 8 requirements):
Ch.6's actual attention weights (`Softmax(QK^T/sqrt(d_k))`) live deep inside
MONAI's SwinTransformer internals (individual WindowAttention blocks, with a
version-specific class path/shape). Reaching into that level of undocumented
internal API is exactly the kind of assumption that already broke this
project once (the img_size incompatibility in src/models/swin_unetr_model.py)
-- so this script deliberately works one level up, at the stable, public
`model.swinViT` attribute (the Swin encoder), using Grad-CAM (Selvaraju et
al., 2017) on its deepest stage output: forward-hook the activation,
backward from the predicted foreground ("spleen") score, weight each channel
by its average gradient, and upsample. If `model.swinViT` doesn't exist on
whatever MONAI version ends up installed, this falls back to plain
input-gradient saliency (Simonyan et al., 2013), which needs nothing but the
model's public forward() and therefore cannot break the same way. Whichever
path actually ran is printed and stamped on every figure and in the saved
manifest, so a figure is never silently mislabeled.

LIMITATIONS (also recorded in reports/experiment_log.md):
- This is an approximation of "where the network's representation is most
  sensitive to the predicted class," not literally the softmax window-
  attention weights Swin UNETR computes internally.
- The Grad-CAM path reads the deepest encoder stage, which is heavily
  downsampled (a 96^3 crop collapses to roughly 3^3 there), so the heatmap
  is coarse by construction, before upsampling smooths it out visually.
- Computed on a single 96^3 crop centered on the ground-truth spleen
  centroid, not the full sliding-window volume the reported Dice/HD95 use
  (re-deriving a whole-volume heatmap would mean reimplementing sliding-
  window stitching for intermediate activations, out of scope here). The
  predicted-mask panel is still the real sliding-window prediction, cropped
  to the same region, so it matches the reported metrics.
- Uses ground-truth labels to center the crop, which is only valid for
  labeled validation cases, not a deployment-time explanation method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.data.dataset import get_datalists
from src.data.transforms import PATCH_SIZE, get_val_transforms
from src.train import build_model
from src.utils.config import TrainConfig


def _foreground_centroid(label: np.ndarray) -> tuple:
    coords = np.argwhere(label > 0)
    if coords.size == 0:
        return tuple(s // 2 for s in label.shape)
    return tuple(int(round(c)) for c in coords.mean(axis=0))


def _crop_slices(volume_shape: tuple, center: tuple, size: tuple) -> tuple:
    """Slices for a `size`-shaped crop centered on `center`, clamped to volume bounds
    (and shrunk per-axis if the volume itself is smaller than `size`)."""
    slices = []
    for c, s, dim in zip(center, size, volume_shape):
        s = min(s, dim)
        start = max(0, min(c - s // 2, dim - s))
        slices.append(slice(start, start + s))
    return tuple(slices)


def _gradcam_or_saliency(model: torch.nn.Module, crop: torch.Tensor) -> tuple:
    """Returns (heatmap as a numpy array shaped like crop's spatial dims, method name)."""
    model.zero_grad(set_to_none=True)
    captured = {}

    def hook(_module, _inp, out):
        captured["value"] = out[-1] if isinstance(out, (list, tuple)) else out

    handle = None
    try:
        handle = model.swinViT.register_forward_hook(hook)
    except AttributeError:
        handle = None

    crop = crop.clone().requires_grad_(True)
    logits = model(crop)
    target = torch.softmax(logits, dim=1)[:, 1].sum()  # total predicted spleen "mass"

    if handle is not None:
        handle.remove()

    if "value" in captured:
        feat = captured["value"]
        grads = torch.autograd.grad(target, feat, retain_graph=False)[0]
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=crop.shape[2:], mode="trilinear", align_corners=False)
        heatmap = cam.squeeze().detach().cpu().numpy()
        method = "grad-cam (model.swinViT deepest stage)"
    else:
        grads = torch.autograd.grad(target, crop, retain_graph=False)[0]
        heatmap = grads.abs().sum(dim=1).squeeze().detach().cpu().numpy()
        method = "input-gradient saliency (fallback: model.swinViT not found)"

    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, method


def _select_cases(eval_results_path: str, explicit_cases: list) -> dict:
    """{"strong": case_id, "average": case_id, "weak": case_id}, chosen by Dice from
    the already-computed evaluation results -- a data-grounded choice, not a guess."""
    if explicit_cases:
        labels = ["case_1", "case_2", "case_3"][: len(explicit_cases)]
        return dict(zip(labels, explicit_cases))

    with open(eval_results_path) as f:
        results = json.load(f)
    ranked = sorted((c for c in results["per_case"] if c["dice"] is not None), key=lambda c: c["dice"])
    return {"weak": ranked[0]["case_id"], "average": ranked[len(ranked) // 2]["case_id"], "strong": ranked[-1]["case_id"]}


def _process_case(model: torch.nn.Module, device: torch.device, case_files: dict) -> dict:
    sample = get_val_transforms()(case_files)
    image = sample["image"].unsqueeze(0).to(device)  # (1,1,D,H,W)
    label = sample["label"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        full_logits = sliding_window_inference(inputs=image, roi_size=PATCH_SIZE, sw_batch_size=4, predictor=model)
    full_pred = torch.argmax(full_logits, dim=1, keepdim=True)

    label_np = label[0, 0].cpu().numpy()
    center = _foreground_centroid(label_np)
    slices = _crop_slices(label_np.shape, center, PATCH_SIZE)

    image_np = image[0, 0].cpu().numpy()[slices]
    label_crop = label_np[slices]
    pred_crop = full_pred[0, 0].cpu().numpy()[slices]
    crop_tensor = image[:, :, slices[0], slices[1], slices[2]]

    heatmap, method = _gradcam_or_saliency(model, crop_tensor)

    mid = image_np.shape[2] // 2
    return {
        "ct_slice": image_np[:, :, mid],
        "gt_slice": label_crop[:, :, mid],
        "pred_slice": pred_crop[:, :, mid],
        "heatmap_slice": heatmap[:, :, mid],
        "method": method,
    }


def _plot_case(panels: dict, case_id: str, tag: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    titles = ["CT slice", "Ground truth", "Prediction", "Attention/importance", "Overlay"]
    axes[0].imshow(panels["ct_slice"], cmap="gray")
    axes[1].imshow(panels["ct_slice"], cmap="gray")
    axes[1].imshow(np.ma.masked_where(panels["gt_slice"] == 0, panels["gt_slice"]), cmap="autumn", alpha=0.6)
    axes[2].imshow(panels["ct_slice"], cmap="gray")
    axes[2].imshow(np.ma.masked_where(panels["pred_slice"] == 0, panels["pred_slice"]), cmap="autumn", alpha=0.6)
    axes[3].imshow(panels["heatmap_slice"], cmap="jet")
    axes[4].imshow(panels["ct_slice"], cmap="gray")
    axes[4].imshow(panels["heatmap_slice"], cmap="jet", alpha=0.5)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{case_id} ({tag}) -- {panels['method']}", fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config: TrainConfig, checkpoint_path: str, eval_results_path: str, output_dir: str, explicit_cases: list) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_files = get_datalists(config.data_dir)
    files_by_case = {Path(f["image"]).stem: f for f in val_files}

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    cases = _select_cases(eval_results_path, explicit_cases)
    manifest = {"run_name": config.run_name, "checkpoint": str(checkpoint_path), "cases": {}}

    for tag, case_id in cases.items():
        if case_id not in files_by_case:
            print(f"Skipping {tag}={case_id}: not found in the validation split")
            continue
        panels = _process_case(model, device, files_by_case[case_id])
        out_path = str(Path(output_dir) / f"{tag}_{case_id}.png")
        _plot_case(panels, case_id, tag, out_path)
        manifest["cases"][tag] = {"case_id": case_id, "method": panels["method"], "figure": out_path}
        print(f"{tag}: {case_id} -> {out_path} ({panels['method']})")

    manifest_path = str(Path(output_dir) / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin UNETR attention/importance explainability figures")
    parser.add_argument("--config", default="configs/swin_unetr_base.yaml")
    parser.add_argument("--checkpoint", default="outputs/swin_unetr_v1/best_model.pt")
    parser.add_argument("--eval-results", default=None, help="Defaults to experiments/<run_name>/eval_results.json")
    parser.add_argument("--output-dir", default=None, help="Defaults to experiments/<run_name>/explainability")
    parser.add_argument(
        "--cases", nargs="*", default=None, help="Explicit case_id list, overrides auto strong/average/weak selection"
    )
    args = parser.parse_args()

    train_config = TrainConfig.from_yaml(args.config)
    eval_results = args.eval_results or f"experiments/{train_config.run_name}/eval_results.json"
    out_dir = args.output_dir or f"experiments/{train_config.run_name}/explainability"
    main(train_config, args.checkpoint, eval_results, out_dir, args.cases)
