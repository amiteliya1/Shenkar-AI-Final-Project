"""Segmentation evaluation metrics.

The Dice coefficient is mathematically the F1-score computed per voxel:
Dice = 2|P n G| / (|P| + |G|) = F1, where P is the predicted foreground voxel
set and G the ground-truth set. This is Ch.3's Precision/Recall/F1 framework
(F1 = 2PR/(P+R)) applied to a per-voxel foreground/background classification
instead of a single prediction -- not a new, unrelated metric.

HD95 (95th-percentile Hausdorff Distance) has no course analogue (Ch.3 only
teaches Precision/Recall/F1/Accuracy) -- it's the Medical Segmentation
Decathlon leaderboard's standard boundary-quality metric, included alongside
Dice because Dice (a volumetric overlap measure) can look fine even when the
predicted boundary/shape is off in ways that matter clinically.

We reuse MONAI's DiceMetric/HausdorffDistanceMetric rather than
reimplementing this computation.
"""

from monai.metrics import DiceMetric, HausdorffDistanceMetric


def make_dice_metric() -> DiceMetric:
    """Mean Dice over the foreground (spleen) class only; background excluded
    since it dominates the volume and would make the score uninformative."""
    return DiceMetric(include_background=False, reduction="mean")


def make_hd95_metric() -> HausdorffDistanceMetric:
    """95th-percentile Hausdorff Distance over the foreground (spleen) class only.
    Undefined (inf) for a case where prediction or ground truth has no foreground
    voxels at all -- callers should handle that when summarizing across cases."""
    return HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
