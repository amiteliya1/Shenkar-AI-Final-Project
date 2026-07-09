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

_(none yet — first entry lands with the Day 3 baseline U-Net run)_
