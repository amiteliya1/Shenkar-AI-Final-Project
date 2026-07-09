"""MONAI transform pipelines for MSD Task09_Spleen CT volumes.

Preprocessing constants below (HU window, target spacing, patch size) reuse
MONAI's own official Spleen-segmentation reference values, themselves derived
from nnU-Net's automatic dataset fingerprinting for this task -- not arbitrary
choices. Patch size / spacing are still candidates for the Day 6 hyperparameter
search if results warrant revisiting them.
"""

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

TARGET_SPACING = (1.5, 1.5, 2.0)  # mm, (x, y, z)
HU_MIN, HU_MAX = -57, 164  # soft-tissue Hounsfield window for spleen
PATCH_SIZE = (96, 96, 96)

_KEYS = ["image", "label"]


def _load_and_normalize() -> list:
    """Steps shared by train and validation: load, orient, resample, window intensities."""
    return [
        LoadImaged(keys=_KEYS),
        EnsureChannelFirstd(keys=_KEYS),
        Orientationd(keys=_KEYS, axcodes="RAS"),
        Spacingd(keys=_KEYS, pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=_KEYS, source_key="image"),
    ]


def get_train_transforms() -> Compose:
    """Training pipeline: shared preprocessing + balanced patch sampling + light augmentation."""
    return Compose(
        _load_and_normalize()
        + [
            RandCropByPosNegLabeld(
                keys=_KEYS,
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=_KEYS, spatial_axis=0, prob=0.5),
            RandFlipd(keys=_KEYS, spatial_axis=1, prob=0.5),
            RandFlipd(keys=_KEYS, spatial_axis=2, prob=0.5),
            RandRotate90d(keys=_KEYS, prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
            EnsureTyped(keys=_KEYS),
        ]
    )


def get_val_transforms() -> Compose:
    """Validation/test pipeline: shared preprocessing only, no cropping or augmentation.

    Full (uncropped) volumes are kept so that evaluation uses MONAI's sliding-window
    inference over the whole scan, matching how the model will actually be used at test time.
    """
    return Compose(_load_and_normalize() + [EnsureTyped(keys=_KEYS)])
