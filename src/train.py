"""Training entrypoint: model + hyperparameters come entirely from a config file,
so the same script trains both the baseline U-Net (Day 3) and Swin UNETR (Day 5).

    python -m src.train --config configs/baseline_unet.yaml

Optimizer/regularization follow Ch.3: Adam, weight decay (L2), Dropout (set
inside the model), and Early Stopping on validation Dice. Underperforming
runs should be analyzed via Ch.3's Bias-Variance tradeoff (train vs. val gap)
in reports/experiment_log.md, not diagnosed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.data import DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from src.data.dataset import get_datasets
from src.data.transforms import PATCH_SIZE
from src.models.unet_baseline import build_unet
from src.utils.config import TrainConfig
from src.utils.logging import RunLogger
from src.utils.metrics import make_dice_metric


def build_model(config: TrainConfig) -> torch.nn.Module:
    if config.model == "unet":
        return build_unet(dropout=config.dropout)
    raise NotImplementedError(f"model '{config.model}' is not implemented yet (Swin UNETR lands on Day 5)")


def train(config: TrainConfig) -> None:
    set_determinism(seed=config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = get_datasets(config.data_dir, num_workers=config.num_workers)
    # RandCropByPosNegLabeld yields `num_samples` patches per volume (a list per
    # __getitem__ call) -- list_data_collate flattens these into one flat batch.
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
    )
    # Validation volumes are kept whole (see transforms.py); one volume per step,
    # sliding_window_inference below handles the internal patch-wise batching.
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=config.num_workers)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    dice_metric = make_dice_metric()
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    experiment_dir = Path("experiments") / config.run_name
    checkpoint_dir = Path("outputs") / config.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(str(experiment_dir))

    best_val_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)

        val_dice = None
        if epoch % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, roi_size=PATCH_SIZE, sw_batch_size=4, predictor=model
                    )
                    val_outputs = [post_pred(x) for x in decollate_batch(val_outputs)]
                    val_labels = [post_label(x) for x in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                val_dice = dice_metric.aggregate().item()
                dice_metric.reset()

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            else:
                epochs_without_improvement += config.val_interval

        logger.log(epoch=epoch, train_loss=epoch_loss, val_dice=val_dice)
        status = f"epoch {epoch}/{config.max_epochs} - loss {epoch_loss:.4f}"
        if val_dice is not None:
            status += f" - val_dice {val_dice:.4f} (best {best_val_dice:.4f})"
        print(status)

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no val_dice improvement for {config.early_stopping_patience} epochs")
            break

    logger.plot()
    print(f"Done. Best val_dice: {best_val_dice:.4f}. Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model on MSD Task09_Spleen")
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/baseline_unet.yaml")
    args = parser.parse_args()
    train(TrainConfig.from_yaml(args.config))
