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
  evaluate.py        Dice / Hausdorff95 evaluation on the validation/test split
  explainability/   Swin UNETR attention extraction and visualization
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
python -m src.train --config configs/swin_unetr_smoketest.yaml
python -m src.evaluate --config configs/swin_unetr_base.yaml --checkpoint outputs/<run_name>/best_model.pt  # Day 7, not implemented yet
```

Each run writes its logs/metrics to `experiments/<run_name>/` and its checkpoint to
`outputs/<run_name>/`.

## Running on Slurm (Shenkar GPU server)

```bash
mkdir -p SlurmLogs        # Slurm does not create missing --output/--error directories
sbatch slurm/train.sbatch configs/baseline_unet.yaml           # or any other config
squeue -u $USER
tail -f SlurmLogs/segmentation_train_<jobid>.out
```
`slurm/train.sbatch` requests the `gpu` partition, 1x NVIDIA L4, and activates `.venv` before
calling `src.train` with whatever config path is passed as its argument (defaults to
`configs/baseline_unet.yaml`) — the same script runs every model, since model choice lives in
the config file, not the script.

## Status

- [x] Day 1: repo scaffold, dataset download script, environment check
- [x] Day 2: data pipeline (transforms, train/val split, sanity-check visualization)
- [x] Day 3: baseline 3D U-Net + training script. First run (Slurm job 316, NVIDIA L4): best
      val Dice 0.4749, early-stopped at epoch 75/100 — see `reports/experiment_log.md`. Notably
      below the ~0.90+ published range for this task; root cause not yet confirmed.
- [x] Day 5: Swin UNETR model implemented; smoke test passed on NVIDIA L4 after fixing an
      `img_size`/MONAI-version incompatibility (see `reports/experiment_log.md`).
      `configs/swin_unetr_base.yaml` (full 100-epoch run, matched to the baseline's methodology)
      ready, not yet run.
- [ ] Day 6 onward: see `reports/experiment_log.md` and the approved project plan
