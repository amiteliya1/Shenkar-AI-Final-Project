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

### swin_unetr_smoketest — Swin UNETR — 2026-07-09 — PASSED (re-run after the img_size fix)

**Config:** `configs/swin_unetr_smoketest.yaml`, Shenkar gpu partition, 1x NVIDIA L4.

**Result:** Completed both epochs without error. Epoch 1/2: loss 1.3294, val_dice 0.0372.
Epoch 2/2: loss 1.1233, val_dice 0.1755. Checkpoint saved at
`outputs/swin_unetr_smoketest/best_model.pt`.

**Analysis:** This confirms the fix worked and the full pipeline (data loading, forward/backward
pass, mixed-precision, sliding-window validation, checkpointing) runs end-to-end on the L4 —
that was this run's only purpose. Loss decreasing and val_dice rising across just 2 epochs is
a good sign the model is learning correctly, but **0.1755 is not comparable to the baseline's
0.4749** — the baseline trained for 75 epochs before early stopping, this ran for 2. Comparing
them directly would be misleading; the real comparison happens after the full run below.

**Decision:** Pipeline confirmed correct. Proceed to a full run with `configs/swin_unetr_base.yaml`
(new), matching `configs/baseline_unet.yaml`'s methodology exactly (Adam, lr=1e-4,
weight_decay=1e-5, dropout=0.2, max_epochs=100, val_interval=5, early_stopping_patience=20,
seed=0, same data/split) so the eventual baseline-vs-Swin-UNETR comparison is fair. The one
deliberate difference is `batch_size=1` (vs. the baseline's 2) — an L4 memory necessity for
Swin UNETR's heavier attention activations at the same patch size, already justified in the
Day 5 pre-training setup note above, not a fairness compromise on the data/methodology side.

### swin_unetr_v1 — Swin UNETR — 2026-07-09 — CANCELLED at wall-clock time limit (Slurm job 322)

**Config:** `configs/swin_unetr_base.yaml`, Shenkar gpu partition, 1x NVIDIA L4, requested
`--time=04:00:00`.

**Result:** `JOB 322 ON slurm-gpu CANCELLED DUE TO TIME LIMIT` at epoch 44/100 (~30 minutes
elapsed). Best val_dice reached before cancellation: **0.5353 at epoch 35** — already above
the baseline's final 0.4749, though the run never got to complete or early-stop on its own
terms, so this isn't yet the real comparison point either.

**Analysis:** Not a modeling problem — the job was cancelled purely on wall-clock time, at 30
minutes, despite `--time=04:00:00` being requested in `slurm/train.sbatch` at the time. That
mismatch indicates something on the cluster (partition or QOS `MaxWallTime`) is silently
capping the effective limit below what's requested, rather than our request being the binding
constraint. `train.py` had no resume capability, so the partial progress could only be
discarded and restarted from epoch 1.

**Decision:** Two changes, both in this repo now: (1) `src/train.py` now saves a full
training-state checkpoint (`outputs/<run_name>/last_checkpoint.pt`: model + optimizer + AMP
scaler + epoch + early-stopping counters) every epoch and automatically resumes from it if
present, so a future time-limit cancellation loses at most one epoch of progress instead of the
whole run; `src/utils/logging.py`'s `RunLogger` now loads any existing `metrics.csv` on
resume so the learning curve stays continuous across the interruption. (2)
`slurm/train.sbatch`'s `--time` raised to 06:00:00 as a best-effort increase, with an explicit
note to verify the partition's actual `MaxTime` since raising the requested number may not be
the real fix if there's a hard cluster-side cap.

**Important:** job 322 itself ran under the *old* code, which never wrote a `last_checkpoint.pt`
— only the weights-only `best_model.pt` (epoch 35) exists for it, which the new resume logic
cannot use (no optimizer/scaler/epoch state). So job 322's 44 epochs of progress cannot be
recovered; resubmitting `sbatch slurm/train.sbatch configs/swin_unetr_base.yaml` restarts
`swin_unetr_v1` from epoch 1, now under the resume-capable code. Any *subsequent* cancellation
of that new run will resume correctly. Same config/dataset/split/val_interval/max_epochs as
before, so the eventual comparison with the baseline is unaffected by this restart.
