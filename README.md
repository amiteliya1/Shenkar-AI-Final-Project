# Medical Image Segmentation with Swin Transformers

Shenkar University neural networks course — final project.

Segments the spleen from abdominal CT scans (MSD Task09_Spleen), comparing a classic 3D U-Net
baseline against a Swin UNETR (vision transformer) model on the same data pipeline, and
visualizing the Swin UNETR's attention maps as an explanation of its predictions.

See `reports/related_work.md` for background on prior segmentation approaches, and
`reports/experiment_log.md` for the experiment-by-experiment process log (failed attempts,
hyperparameter search, analysis) required by the course.

## Project layout

```
data/               downloaded MSD Task09_Spleen (gitignored, see Setup below)
src/
  data/             dataset loading + MONAI transforms
  models/           baseline 3D U-Net and Swin UNETR wrappers
  train.py          training entrypoint (model + hyperparameters selected via a config file)
  evaluate.py       Dice / HD95 evaluation of a checkpoint on the validation split
  explainability/   Swin UNETR Grad-CAM/saliency explainability (attention_map.py)
  utils/            metrics, logging, config loading
configs/            one YAML file per experiment run
experiments/        logged results (CSV) and plots per run
reports/            related_work.md, experiment_log.md
notebooks/          exploratory notebooks and figures for the presentation
outputs/            checkpoints and predictions (gitignored)
slurm/              sbatch scripts for running on the Shenkar GPU server
```

## Setup (on the university cluster)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset:

```bash
python -m src.data.download
```

This uses MONAI's `DecathlonDataset` to fetch and extract Task09_Spleen into `data/`. If the
Google Drive download is rate-limited (common for MSD), download `Task09_Spleen.tar` manually
from the [MSD dataset page](http://medicaldecathlon.com/) and extract it into `data/Task09_Spleen/`.

Verify the environment before training:

```bash
python -m src.utils.env_check
```

Sanity-check the data pipeline (confirms orientation/spacing/windowing are correct before
trusting any training run):

```bash
python -m src.data.visualize_sample --data-dir data --index 0
```

## Running an experiment

```bash
python -m src.train --config configs/baseline_unet.yaml
python -m src.train --config configs/swin_unetr_base.yaml
```

Each run writes its logs/metrics to `experiments/<run_name>/` and its checkpoint to
`outputs/<run_name>/`. If a run is interrupted (e.g. a Slurm time-limit cancellation),
resubmitting the exact same command resumes automatically from
`outputs/<run_name>/last_checkpoint.pt` instead of restarting from epoch 1.

## Evaluating a trained model

```bash
python -m src.evaluate --config configs/baseline_unet.yaml --checkpoint outputs/baseline_unet_v1/best_model.pt
python -m src.evaluate --config configs/swin_unetr_base.yaml --checkpoint outputs/swin_unetr_v1/best_model.pt
```
Computes mean Dice and HD95 (95th-percentile Hausdorff Distance) on the same validation split
used during training (`src/data/dataset.py`'s fixed-seed 80/20 split — evaluation does not
create a different one), for both the baseline and Swin UNETR, so the two are directly
comparable. Writes per-case and summary metrics as JSON to `experiments/<run_name>/eval_results.json`
by default (override with `--output`).

## Explainability (Swin UNETR)

```bash
python -m src.explainability.attention_map \
    --config configs/swin_unetr_base.yaml --checkpoint outputs/swin_unetr_v1/best_model.pt
```
Generates a 5-panel figure (CT slice, ground truth, prediction, attention/importance heatmap,
overlay) for the strong/average/weak validation cases, auto-selected by Dice from
`experiments/swin_unetr_v1/eval_results.json`. Saved to `experiments/<run_name>/explainability/`,
alongside a `manifest.json` recording which case got which label and which explanation method
actually ran. **Method and limitations:** see the module docstring in
`src/explainability/attention_map.py` and `reports/experiment_log.md` — briefly, this is
Grad-CAM on the Swin encoder's deepest stage (an approximation of "what mattered," not the raw
window-attention weights themselves), with an automatic fallback to input-gradient saliency if
the installed MONAI version doesn't expose the expected internal attribute.

## Running on Slurm (Shenkar GPU server)

```bash
mkdir -p SlurmLogs        # Slurm does not create missing --output/--error directories
sbatch slurm/train.sbatch configs/baseline_unet.yaml           # or any other training config
sbatch slurm/evaluate.sbatch configs/baseline_unet.yaml outputs/baseline_unet_v1/best_model.pt
sbatch slurm/explain.sbatch                                     # defaults match swin_unetr_v1
squeue -u $USER
tail -f SlurmLogs/segmentation_train_<jobid>.out   # or _eval_/_explain_
```
`slurm/train.sbatch`, `slurm/evaluate.sbatch`, and `slurm/explain.sbatch` all request the `gpu`
partition, 1x NVIDIA L4, and activate `.venv` before calling the corresponding `src` module with
whatever arguments are passed to `sbatch` — the same scripts run every model/checkpoint, since
model choice lives in the config file, not the script.

## Status

- [x] Day 1: repo scaffold, dataset download script, environment check
- [x] Day 2: data pipeline (transforms, train/val split, sanity-check visualization)
- [x] Day 3: baseline 3D U-Net + training script. First run (Slurm job 316, NVIDIA L4): best
      val Dice 0.4749, early-stopped at epoch 75/100 — see `reports/experiment_log.md`. Notably
      below the ~0.90+ published range for this task; root cause not yet confirmed.
- [x] Day 5: Swin UNETR model implemented and fully trained. Job 322 (pre-resume-support code)
      was cancelled by a Slurm time limit at epoch 44/100; after adding checkpoint-resume
      support, the restarted `swin_unetr_v1` run completed, early-stopping at epoch 55 with
      best val Dice **0.5353** — above the baseline's 0.4749, though not yet HD95-compared.
      See `reports/experiment_log.md` for the full sequence.
- [x] Day 7: `src/evaluate.py` added and run for both models on the shared validation split.
      **Final comparison:** baseline mean Dice 0.4750 / mean HD95 154.3mm vs. Swin UNETR mean
      Dice 0.5353 / mean HD95 156.0mm — Swin UNETR wins on Dice, HD95 is essentially tied (and
      more variable for Swin UNETR). Both far below the ~0.90+ / single-digit-mm published
      range for this task. Full analysis and conclusions in `reports/experiment_log.md`.
- [x] Day 8: `src/explainability/attention_map.py` added (Grad-CAM on the Swin encoder's
      deepest stage, with an input-gradient-saliency fallback) and run on the strong/average/weak
      validation cases (`spleen_12`/`spleen_10`/`spleen_25`) — all three used Grad-CAM, the
      saliency fallback never triggered. The weak case visually confirms the Day 7 hypothesis
      that Swin UNETR's HD95 blowups come from stray false-positive blobs, not boundary error;
      the average case is a clean example of the heatmap tracking the true organ. Figures and
      `manifest.json` in `experiments/swin_unetr_v1/explainability/`; full analysis in
      `reports/experiment_log.md`.
- [ ] Day 9 onward: see `reports/experiment_log.md` and the approved project plan
