# Experiment Log

Running record of every training run: what we tried, what happened, why, and what changed next.
This is the primary evidence for the course's "show the process" requirement — filled in as we
go, not reconstructed afterwards.

Each entry should cover: config used, result, analysis, and the decision it led to.

## Template for each entry

```
### <run_name> — <model> — <date>

**Config:** (link to configs/<file>.yaml, or key hyperparameters inline)

**Result:** (Dice / loss curves / qualitative observations)

**Analysis:** (why did it do what it did — underfitting? overfitting? class imbalance?
                bad patch sampling? learning rate too high/low?)

**Decision:** (what changes as a result — next config, or "keep as final")
```

---

## Pre-training setup (Day 3, before the first run)

Baseline architecture and starting hyperparameters are finalized in
`src/models/unet_baseline.py` / `configs/baseline_unet.yaml`, reusing course (Ch.3/Ch.4/Ch.10)
and MONAI-tutorial defaults rather than guessing:

- **Model:** MONAI `UNet`, 5 levels, channels (16,32,64,128,256), `num_res_units=2`,
  **InstanceNorm** (not Ch.3's default BatchNorm — 3D patches force small batch sizes, where
  BatchNorm statistics are unreliable), `dropout=0.2` (Ch.3 regularization).
- **Optimizer:** Adam, `lr=1e-4`, `weight_decay=1e-5` (L2, Ch.3).
- **Loss:** `DiceCELoss` (Dice + Cross-Entropy combined) — Dice directly optimizes the overlap
  metric we report (see Dice = F1-per-voxel note in `src/utils/metrics.py`), Cross-Entropy adds
  a per-voxel classification signal that's less flat early in training than Dice alone.
- **Early stopping:** patience of 20 epochs on validation Dice (Ch.3).
- **Batch composition:** `batch_size=2` volumes x `num_samples=4` patches per volume
  (`RandCropByPosNegLabeld`) = 8 patches of 96^3 per training step.

These are starting values, not final ones — to be revisited here with results once the first
run completes.

## Entries

### baseline_unet_v1 — 3D U-Net (MONAI, InstanceNorm, num_res_units=2) — 2026-07-09

**Config:** `configs/baseline_unet.yaml` (Adam, lr=1e-4, weight_decay=1e-5, dropout=0.2,
DiceCELoss, batch_size=2 x 4 patches of 96^3, early_stopping_patience=20). Run on Shenkar's
gpu partition, 1x NVIDIA L4, Slurm job 316.

**Result:** Early-stopped at epoch 75/100 (no val_dice improvement for the configured 20-epoch
patience). Best validation Dice = **0.4749**, reached around epoch 55. Checkpoint saved at
`outputs/baseline_unet_v1/best_model.pt`.

**Analysis (provisional — see open questions below):** 0.4749 is well below the ~0.90+ Dice
that MONAI's own reference U-Net tutorial and nnU-Net report on this exact task (Task09_Spleen)
with very similar architecture/hyperparameters. In Ch.3's Bias-Variance framing: plateauing for
20 straight epochs after epoch 55 looks like the model settled into a ceiling rather than still
slowly improving, which is consistent with either (a) high bias (optimization or model capacity
genuinely stuck, e.g. learning rate too low/no LR schedule to escape a plateau) or (b) a
pipeline defect that caps achievable Dice regardless of training time (e.g. a label/prediction
channel mismatch, or an unverified data-orientation issue) rather than a hyperparameter problem
per se. Distinguishing (a) from (b) requires the actual train-loss curve alongside val-Dice
(not yet available on this machine — `experiments/baseline_unet_v1/metrics.csv` and
`learning_curve.png` live only on the Shenkar server) and confirmation that the Day 2
`visualize_sample.py` sanity check was actually inspected and looked correct.

**Decision:** Proceeding to Swin UNETR (Day 5) on this baseline result as-is, by explicit
decision, without having confirmed via `metrics.csv`/`learning_curve.png` whether the low Dice
is premature early stopping or a pipeline issue. **Open risk carried forward:** since both
models share the same data pipeline, if the root cause turns out to be a pipeline defect, the
Swin UNETR numbers will inherit the same ceiling and the comparison between the two will still
need revisiting. Flagging this explicitly here so it isn't lost.

## Pre-training setup (Day 5, before the Swin UNETR smoke test)

`src/models/swin_unetr_model.py` / `configs/swin_unetr_smoketest.yaml`:

- **Model:** MONAI `SwinUNETR`, `feature_size=24` (MONAI's default, lighter than the 48 used for
  BraTS in the original paper), **InstanceNorm** (same small-batch rationale as the baseline),
  `use_checkpoint=True` (activation checkpointing — trades compute for memory).
- **Same data pipeline as the baseline:** identical `src/data/transforms.py` (96^3 patches, same
  HU window/spacing) and identical fixed-seed 80/20 split (`src/data/dataset.py`) — the encoder
  is the only thing that differs between the two runs.
- **Memory vs. baseline:** `batch_size=1` (vs. the baseline's 2) purely as an L4 headroom
  safeguard for Swin UNETR's heavier attention activations at the same patch size; `train.py`
  now also uses mixed precision (autocast + GradScaler) for both models, added for the same
  memory reason (does not affect the already-saved baseline_unet_v1 result).
- **Smoke test only:** `max_epochs=2` — purpose is to confirm the run fits in L4 memory and
  completes end-to-end, not to produce a comparable result yet.

### swin_unetr_smoketest — Swin UNETR — 2026-07-09 — FAILED at model init (Slurm job 319)

**Config:** `configs/swin_unetr_smoketest.yaml`, Shenkar gpu partition, 1x NVIDIA L4.

**Result:** Data loading succeeded (train + val datasets built correctly). Crashed immediately
at model construction:
```
TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'
```

**Analysis:** Not a data-pipeline or hyperparameter issue — the installed MONAI release on the
server has fully removed the `img_size` constructor argument (newer `SwinUNETR` versions are
input-size-agnostic at construction time; relative position bias is window-size-based, not
tied to a fixed image size). Root cause: `requirements.txt` pins only a lower bound
(`monai[nibabel,tqdm]>=1.3`), so pip installed a newer MONAI release than the API `img_size`
usage assumed, on the server. This was a version-compatibility bug in
`src/models/swin_unetr_model.py`, not a modeling or fairness problem — nothing about the data
split, transforms, or the baseline result is implicated.

**Decision:** Removed `img_size` from the `SwinUNETR(...)` call in `src/models/swin_unetr_model.py`
(and the now-unused `PATCH_SIZE` import). The 96^3 patch size is still enforced identically for
both models from the data/inference side (`src/data/transforms.py`, `train.py`'s
`sliding_window_inference(roi_size=PATCH_SIZE, ...)`), so the fair-comparison property is
unaffected. Re-run `configs/swin_unetr_smoketest.yaml` next.
