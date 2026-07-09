"""Segmentation evaluation metric.

The Dice coefficient is mathematically the F1-score computed per voxel:
Dice = 2|P n G| / (|P| + |G|) = F1, where P is the predicted foreground voxel
set and G the ground-truth set. This is Ch.3's Precision/Recall/F1 framework
(F1 = 2PR/(P+R)) applied to a per-voxel foreground/background classification
instead of a single prediction -- not a new, unrelated metric.

We reuse MONAI's DiceMetric rather than reimplementing this computation.
"""

from monai.metrics import DiceMetric


def make_dice_metric() -> DiceMetric:
    """Mean Dice over the foreground (spleen) class only; background excluded
    since it dominates the volume and would make the score uninformative."""
    return DiceMetric(include_background=False, reduction="mean")
