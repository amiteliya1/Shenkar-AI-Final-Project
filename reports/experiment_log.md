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

### swin_unetr_v1 — Swin UNETR — 2026-07-10 — COMPLETED (restart, resume-capable code)

**Config:** `configs/swin_unetr_base.yaml`, same as job 322 — restarted from epoch 1 (job 322's
progress could not be recovered, see above).

**Result:** Early-stopped at epoch 55/100. Best validation Dice = **0.5353**. Checkpoint saved
at `outputs/swin_unetr_v1/best_model.pt`.

**Analysis:** Best Dice (0.5353) matches job 322's best-before-cancellation value exactly, and
55 = 35 + 20 (`early_stopping_patience`) — consistent with the restart reproducing job 322's
training dynamics deterministically up to the point job 322 was cut off (same seed, same data,
same model, and the added checkpoint-saving code doesn't consume RNG state), then continuing
past epoch 44 to plateau and trigger early stopping at epoch 55. This is a useful reproducibility
check in its own right (Ch.3-relevant: same seed really does reproduce the same run). Swin
UNETR's 0.5353 is above the baseline U-Net's 0.4749, but **both remain far below the ~0.90+
published range for this task** — the open question from the baseline's entry (premature
stopping vs. a shared pipeline issue) is still unresolved and now applies to both models, since
they share the same data pipeline.

**Decision:** Both models have a completed, checkpointed run under matched methodology. Next:
`src/evaluate.py` (Day 7) to get Dice + HD95 on the shared validation split for a real
apples-to-apples comparison — mean training-time val_dice from early stopping is a reasonable
proxy but HD95 and the finalized per-case numbers are the actual comparison artifact for the
report.

## Day 7: evaluation script added — 2026-07-10

`src/evaluate.py` implemented: loads a checkpoint (`--checkpoint`) for the model/data specified
by a training config (`--config`), evaluates it on `src/data/dataset.py`'s exact validation
split (no new split constructed) using the same preprocessing (`get_val_transforms()`) and the
same `sliding_window_inference` settings training used, computing per-case and mean Dice + HD95
(`src/utils/metrics.py`'s new `make_hd95_metric()`). Writes per-case and summary metrics as JSON
to `experiments/<run_name>/eval_results.json` by default. `slurm/evaluate.sbatch` added to run
it on the cluster (30-minute time limit — evaluation is forward-passes only, far cheaper than
training).

## Final comparison: baseline 3D U-Net vs. Swin UNETR — 2026-07-10

Source: `experiments/baseline_unet_v1/eval_results.json`, `experiments/swin_unetr_v1/eval_results.json`
(both evaluated on the identical 8-case validation split by `src/evaluate.py`).

| Model | Mean Dice | Std Dice | Mean HD95 (mm) | Std HD95 |
|---|---|---|---|---|
| Baseline 3D U-Net | 0.4750 | 0.2374 | 154.25 | 28.10 |
| Swin UNETR        | 0.5353 | 0.1569 | 155.98 | 62.79 |

**Dice:** Swin UNETR improves mean Dice by +0.060 (~13% relative) and is also more *consistent*
across cases (std 0.157 vs. 0.237). Looking at per-case Dice, this is driven largely by rescuing
the baseline's two worst cases: `spleen_41` (0.141 -> 0.601, +0.46) and `spleen_44`
(0.157 -> 0.416, +0.26) — plausibly the baseline CNN's limited receptive field failing on an
atypically-positioned/shaped spleen, where Swin UNETR's window-based global attention does
better. Against that, Swin UNETR regresses on a few cases the baseline handled reasonably well
(`spleen_25`: 0.458 -> 0.313, `spleen_28`: 0.471 -> 0.355).

**HD95 is essentially flat (154 vs. 156mm) despite the Dice gain, and its variance more than
doubled (std 28 vs. 63mm) — this is the key finding Dice alone would have hidden.** Both
absolute numbers are enormous for an organ the size of a spleen (~10cm) — single-digit-to-low-
double-digit mm HD95 is the published range for this task, so ~155mm indicates both models are
placing at least some predicted voxels very far from the true spleen somewhere in the volume,
not merely drawing an imperfect boundary around it. Most tellingly: on the exact two cases where
Swin UNETR's Dice improved the most (`spleen_41`, `spleen_44`), its HD95 got *worse* than the
baseline's (194 -> 257mm, 185 -> 255mm). Better overlap and worse worst-point distance on the
same case is consistent with Swin UNETR predicting most of the true spleen correctly (driving
Dice up) while also predicting one or more small false-positive blobs elsewhere in the
abdomen (barely denting Dice, since Dice measures overlap in volume, but dominating HD95, which
is sensitive to the single worst-matched surface point). This is precisely why HD95 was added
alongside Dice in the first place (see `src/utils/metrics.py`) — a purely volumetric metric can
look like a straightforward improvement while masking a localization failure a boundary metric
catches immediately.

**Conclusions:**
1. Swin UNETR is the better model on this comparison by mean Dice, but the improvement is
   partial and uneven, not a clean win — it trades some regressed cases for large rescues on
   the baseline's worst failures.
2. Neither model is close to the published ~0.90+ Dice / single-digit-mm HD95 range for
   Task09_Spleen — both should be read as "which one fails less badly," not as a good result in
   absolute terms. The open question from the baseline's entry above (premature early stopping
   vs. a shared pipeline issue) is still unresolved for both models.
3. New, data-grounded hypothesis for follow-up: the HD95 blowups look like stray false-positive
   predictions outside the true spleen region rather than a poorly-shaped boundary around a
   correctly-located spleen. Worth visualizing `spleen_41`/`spleen_44`'s predictions directly to
   confirm, and considering a largest-connected-component post-processing step (standard
   practice in medical segmentation specifically for this failure mode) as a candidate fix.

## Day 8: explainability method — 2026-07-10

`src/explainability/attention_map.py` implemented, generating a 5-panel figure (CT slice,
ground truth, prediction, attention/importance heatmap, overlay) for the strong/average/weak
Swin UNETR validation cases (chosen by Dice directly from `eval_results.json` — a data-grounded
selection, not a guess; the "weak" and "strong" picks may well land on `spleen_41`/`spleen_44`
from the finding above, which would make these figures a direct visual check of that hypothesis).

**Method — chosen for safety, not just novelty:** Ch.6's actual attention weights
(`Softmax(QK^T/sqrt(d_k))`) live inside MONAI's `SwinTransformer` internals (individual
`WindowAttention` blocks), at a version-specific depth. Reaching into undocumented internals at
that level is exactly what already broke this project once (the `img_size` incompatibility in
`src/models/swin_unetr_model.py`, Day 5) — a second such break wouldn't be discovered until
after a wasted Slurm submission, same as before. So this script works one level up instead: a
forward hook on the stable, public `model.swinViT` attribute (the Swin encoder), producing a
Grad-CAM heatmap (Selvaraju et al., 2017) from its deepest stage — weight each output channel
by its average gradient w.r.t. the predicted spleen score, sum, ReLU, upsample. If
`model.swinViT` doesn't exist on whatever MONAI version is installed, the script automatically
falls back to plain input-gradient saliency (Simonyan et al., 2013), which needs nothing but
the model's public `forward()` and therefore cannot break the same way. Whichever path actually
fires is printed and stamped on every figure's title and in a saved `manifest.json`, so a result
is never silently mislabeled as the other method.

**Limitations (by design, not oversights):**
- An approximation of "where the network's representation is most sensitive to the predicted
  class," not the literal window-attention weights Swin UNETR computes internally.
- The Grad-CAM path reads the deepest encoder stage, which is heavily downsampled (the 96^3
  crop collapses to roughly 3^3 there) — coarse by construction, before upsampling smooths it.
- Computed on a single 96^3 crop centered on the ground-truth spleen centroid, not the full
  sliding-window volume the reported Dice/HD95 use (re-deriving a whole-volume heatmap would
  mean reimplementing sliding-window stitching for intermediate activations). The prediction
  panel is still the real sliding-window prediction, cropped to the same region, so it matches
  the reported metrics.
- Uses the ground-truth label to center the crop — valid for labeled validation cases, not a
  deployment-time explanation method.

### Explainability run — 2026-07-10 — COMPLETED (Slurm, `slurm/explain.sbatch`)

**Result:** All three cases ran the primary Grad-CAM path (`model.swinViT` was present on the
installed MONAI version — the saliency fallback never triggered), per `manifest.json`:

| Label | Case | Dice | HD95 (mm) | Method |
|---|---|---|---|---|
| weak | spleen_25 | 0.313 | 121.8 | grad-cam |
| average | spleen_10 | 0.593 | 127.5 | grad-cam |
| strong | spleen_12 | 0.765 | 98.7 | grad-cam |

**Analysis:**

1. **Auto-selection picks by Dice, not HD95 — so it missed the cases the Day 7 finding actually
   flagged.** The weak/average/strong logic selects the min/max/closest-to-mean *Dice* case from
   `eval_results.json`. `spleen_41` and `spleen_44` — the two cases Day 7 singled out for the
   HD95 blowup (Dice went up, HD95 got far worse: 194→257mm and 185→255mm) — are Dice 0.601 and
   0.416 respectively, unremarkable enough on Dice alone that neither was auto-selected here.
   This is a real limitation of Dice-only case selection, not a bug: it answers "show me a
   typical success/failure," not "show me the case with the worst boundary error." Worth a
   manual follow-up (see Decision below).

2. **The weak case (spleen_25) independently confirms the "stray false-positive blob"
   hypothesis from Day 7 — on a different case than originally guessed.** Its prediction panel
   shows the model correctly covering the left portion of the true spleen, *plus* a second,
   spatially separate red region with no corresponding ground-truth label at all. That's a
   direct visual instance of the exact failure mode inferred indirectly from the Dice/HD95 gap
   in the Day 7 analysis (good volumetric overlap coexisting with a disconnected false-positive
   region that a boundary metric would punish hard) — now confirmed by eye, and evidently not
   confined to `spleen_41`/`spleen_44` alone.

3. **The average case (spleen_10) is the cleanest evidence the explainability method is doing
   its job.** The Grad-CAM heatmap's single hot region sits directly over the true spleen body,
   and the prediction closely tracks the ground-truth boundary in this slice. This is the figure
   to lead with in the presentation as "the model is attending to the right organ."

4. **The strong case (spleen_12, highest Dice of the three) visually undershoots the ground
   truth in this slice** — the prediction misses a substantial chunk of the crescent at both
   ends, despite having the best *volumetric* Dice (0.765) of the three. Reminder that a single
   2D slice isn't representative of full 3D overlap (the reported Dice is over the whole volume),
   and that "strong by Dice" doesn't guarantee a visually complete mask in every slice. The
   heatmap's hot region concentrates on the organ's core and cools toward the boundary, roughly
   consistent with the under-segmentation being a boundary-confidence issue rather than the model
   losing track of the organ.

5. **All three heatmaps are smooth, low-resolution blobs, as anticipated in the module
   docstring** — expected, since the deepest Swin stage collapses the 96^3 crop to roughly 3^3
   before upsampling. They're good enough to answer "is the model looking at roughly the right
   region" (yes, in all three cases) but not to explain slice-precise boundary decisions — this
   matches the limitation already documented before the run, not a surprise found after it.

**Decision:** Figures and `manifest.json` committed under `experiments/swin_unetr_v1/explainability/`.
The Day 7 false-positive-blob hypothesis is now visually supported (point 2 above), which
strengthens the case for trying largest-connected-component postprocessing as a follow-up fix —
still the leading candidate from Day 7, now with two independent cases (`spleen_25` here,
`spleen_41`/`spleen_44` from Day 7) suggesting the same failure mode rather than one. A targeted
re-run of `attention_map.py` specifically on `spleen_41`/`spleen_44` (bypassing the auto Dice-based
selection) would let us see whether *their* false-positive blobs are visually similar to
`spleen_25`'s, but is not required to move forward — the postprocessing experiment can be
evaluated on Dice/HD95 directly regardless of whether we visualize those two specific cases first.

## Pre-experiment setup (Day 9, before the postprocessing run)

`src/evaluate.py --postprocess` added: applies MONAI's `KeepLargestConnectedComponent`
(`src/utils/metrics.py`'s new `make_largest_component_postprocess()`) to each case's argmaxed
prediction before scoring, discarding every connected foreground region except the single
largest one. Purely post-hoc — no retraining, same checkpoint (`outputs/swin_unetr_v1/best_model.pt`),
same validation split, same Dice/HD95 metrics — so it is a clean apples-to-apples "does this fix
change the numbers" comparison against the existing `experiments/swin_unetr_v1/eval_results.json`.
Writes to `experiments/swin_unetr_v1/eval_results_postprocessed.json` by default (never overwrites
the raw result). Also wired into `slurm/evaluate.sbatch` via an optional 4th `postprocess` arg.

**Hypothesis being tested:** if the Day 7/Day 8 finding is right — that Swin UNETR's high HD95
(155.98mm mean, worse than the baseline's 154.25mm despite a Dice win) comes from stray,
spatially disconnected false-positive blobs rather than a poorly-shaped boundary around a
correctly-located spleen — then discarding every predicted component except the largest should
substantially reduce HD95 (removing the far-away false-positive that dominates the worst-point
distance) while leaving Dice roughly unchanged or only slightly lower (the discarded blobs are
volumetrically small, per the Day 7 analysis, or Dice would already have been low on those
cases). A result that *doesn't* show this pattern would suggest the true cause is something else
(e.g. a genuinely misshapen boundary on the main component) and the false-positive-blob
hypothesis needs revisiting.

### swin_unetr_v1 + largest-connected-component postprocessing — 2026-07-10 — COMPLETED

**Config:** `python -m src.evaluate --config configs/swin_unetr_base.yaml --checkpoint
outputs/swin_unetr_v1/best_model.pt --postprocess` (via `slurm/evaluate.sbatch ... postprocess`).
Same checkpoint, split, and metrics as the raw `eval_results.json` — postprocessing only.

**Result:** Mean Dice **0.5353 → 0.7649** (+0.230, +43% relative). Mean HD95 **155.98mm → 18.46mm**
(-137.5mm, -88% relative). Every one of the 8 validation cases improved on *both* metrics
simultaneously — no regressions:

| Case | Dice (raw→pp) | HD95mm (raw→pp) |
|---|---|---|
| spleen_19 | 0.593 → 0.917 (+0.325) | 144.1 → 7.0 (-137.1) |
| spleen_28 | 0.355 → 0.534 (+0.179) | 124.3 → 51.2 (-73.0) |
| spleen_13 | 0.646 → 0.927 (+0.280) | 120.1 → 4.6 (-115.5) |
| spleen_41 | 0.601 → 0.836 (+0.235) | 256.8 → 5.1 (-251.7) |
| spleen_10 | 0.593 → 0.870 (+0.277) | 127.5 → 6.8 (-120.8) |
| spleen_44 | 0.416 → 0.646 (+0.230) | 254.7 → 15.2 (-239.5) |
| spleen_12 | 0.765 → 0.827 (+0.061) | 98.7 → 10.8 (-87.9) |
| spleen_25 | 0.313 → 0.562 (+0.249) | 121.8 → 47.1 (-74.7) |

**Analysis:** This is about as clean a confirmation of the false-positive-blob hypothesis as the
data could give. `spleen_41` and `spleen_44` — the two cases Day 7 flagged for the HD95 blowup —
get by far the largest HD95 drops (-251.7mm, -239.5mm), collapsing from "worst in the set" to
single-digit mm, consistent with their error being almost entirely one or more stray blobs far
from the true spleen rather than a poorly-shaped boundary. `spleen_25` — the Day 8 explainability
"weak" case whose prediction panel visibly showed a disconnected red region with no ground-truth
counterpart — also improves substantially on both metrics (+0.249 Dice, -74.7mm HD95), which is
exactly what removing that visible blob should do. That every case (not just these three)
improves on both metrics simultaneously indicates spurious small components were a pervasive
issue across the whole validation set, not a one-off failure on a couple of hard cases. The
post-postprocessing HD95 (18.46mm mean) is also now in a plausible range relative to the
~single-digit-to-low-double-digit mm published for this task, versus ~155mm before — most of the
gap to published numbers documented as an open question since Day 3 turns out to be explained by
this specific, fixable postprocessing gap rather than a deeper modeling or pipeline defect.

**Decision:** Adopt largest-connected-component postprocessing for Swin UNETR's reported numbers
going forward — it is a strict improvement with no observed downside on this validation set, and
directly targets a failure mode independently confirmed via the eval metrics (Day 7), a
qualitative figure (Day 8), and now this controlled before/after comparison. **Open item before
this becomes the final headline comparison:** the baseline 3D U-Net's raw HD95 (154.25mm) is
essentially as bad as Swin UNETR's pre-postprocessing number, which raises the same question for
it — is the baseline *also* producing stray false-positive components that this same fix would
clean up? Postprocessing has only been evaluated on Swin UNETR so far; applying it unevenly (one
model postprocessed, one not) would break the project's fair-comparison discipline maintained
since Day 5. Next: run `--postprocess` on `baseline_unet_v1` too before updating the final
comparison table in this log or in any report/presentation numbers.

### baseline_unet_v1 + largest-connected-component postprocessing — 2026-07-11 — COMPLETED

**Config:** `python -m src.evaluate --config configs/baseline_unet.yaml --checkpoint
outputs/baseline_unet_v1/best_model.pt --postprocess` (via `slurm/evaluate.sbatch ...
postprocess`). Same checkpoint, split, and metrics as the raw `eval_results.json` — postprocessing
only, evaluated the same way as the Swin UNETR run above so the two stay comparable.

**Result:** Mean Dice **0.4750 → 0.6907** (+0.216, +45% relative). Mean HD95 **154.25mm → 33.90mm**
(-120.4mm, -78% relative). 7 of 8 cases improved on both metrics; one case (`spleen_44`)
regressed on Dice:

| Case | Dice (raw→pp) | HD95mm (raw→pp) |
|---|---|---|
| spleen_19 | 0.609 → 0.905 (+0.296) | 158.9 → 2.8 (-156.1) |
| spleen_28 | 0.471 → 0.845 (+0.374) | 125.4 → 4.2 (-121.1) |
| spleen_13 | 0.681 → 0.894 (+0.214) | 122.8 → 6.4 (-116.4) |
| spleen_41 | 0.141 → 0.634 (+0.493) | 194.1 → 17.8 (-176.3) |
| spleen_10 | 0.460 → 0.845 (+0.385) | 175.9 → 6.7 (-169.2) |
| spleen_44 | 0.157 → **0.000 (-0.157)** | 184.9 → 175.5 (-9.4) |
| spleen_12 | 0.823 → 0.919 (+0.096) | 138.3 → 3.3 (-135.0) |
| spleen_25 | 0.458 → 0.484 (+0.026) | 133.6 → 54.4 (-79.3) |

**Analysis:** In aggregate this replicates the Swin UNETR finding — most of the baseline's huge
raw HD95 was also stray disconnected components, not boundary error, and discarding all but the
largest predicted component fixes most of it. But unlike Swin UNETR (where all 8 cases improved
on both metrics with no downside), `spleen_44` here is a genuine regression: Dice drops from a
already-poor 0.157 all the way to 0.0, and HD95 barely moves (184.9 → 175.5mm, still enormous).
The only way Dice can hit exactly 0.0 after keeping just the largest component is if that
component doesn't overlap the true spleen at all — i.e. for this case the model's *biggest*
predicted blob is itself a false positive, and the (smaller) piece that actually overlapped the
true spleen was the one thrown away. This is the mirror image of the failure mode the fix targets:
"keep the largest component" is only a safe heuristic when the largest component is usually the
correct structure, which held for every Swin UNETR case but not for all baseline cases. It doesn't
overturn the aggregate result (0.4750→0.6907 Dice, 154.25→33.90mm HD95 is still a large, real
improvement driven by 7/8 cases), but it means the fix isn't unconditionally safe and should be
reported as such rather than glossed over.

**Decision:** Adopt postprocessing for the baseline's reported numbers too, so both models are
evaluated identically for the final comparison (the project's fair-comparison discipline
maintained since Day 5). Report the `spleen_44` regression explicitly wherever these numbers are
used — it's a real, informative limitation of the fix, not a rounding artifact.

## Final comparison (postprocessed): baseline 3D U-Net vs. Swin UNETR — 2026-07-11

Source: `experiments/baseline_unet_v1/eval_results_postprocessed.json`,
`experiments/swin_unetr_v1/eval_results_postprocessed.json` — both models, same validation split,
same postprocessing (`KeepLargestConnectedComponent`), so this is the apples-to-apples final
number the project set out to produce.

| Model | Mean Dice | Std Dice | Mean HD95 (mm) | Std HD95 |
|---|---|---|---|---|
| Baseline 3D U-Net (postprocessed) | 0.6907 | 0.3185 | 33.90 | 59.77 |
| Swin UNETR (postprocessed)        | 0.7649 | 0.1594 | 18.46 | 19.28 |

**Conclusions:**
1. Postprocessing closes most of the gap to the published range for both models (Dice moved from
   "far below" to within reach of it; HD95 moved from ~150mm to 18-34mm), confirming the Day
   7/8/9 diagnosis — the dominant failure mode for both architectures was stray disconnected
   false-positive components, not fundamentally wrong segmentations.
2. Applied evenly, Swin UNETR is still the better model on *both* metrics after postprocessing:
   +0.0742 Dice (~11% relative) and -15.44mm HD95 (~46% relative) versus the baseline. The
   pre-postprocessing story (Swin wins Dice, ties on HD95) does not hold once both are
   postprocessed — Swin now wins clearly on both axes.
3. Swin UNETR's postprocessed std HD95 (19.28) is also much tighter than the baseline's (59.77),
   which is inflated by `spleen_44`'s regression (175.5mm, by far the largest remaining error in
   either model's postprocessed results) — a second, independent sign that Swin UNETR's
   underlying predictions are more reliably localized even before this fix is applied.
4. The one caveat: postprocessing is not unconditionally safe (see `spleen_44` above) — it
   assumes the largest predicted component is usually the correct structure, which held for all 8
   Swin UNETR cases but not for the baseline's worst case. Any deployment of this fix should keep
   that failure mode in mind, e.g. by flagging cases where the discarded components are
   volumetrically large relative to the kept one.
