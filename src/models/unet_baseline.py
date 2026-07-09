"""Baseline 3D U-Net.

Course framing (Ch.10 - Autoencoders): this is an Encoder -> Bottleneck ->
Decoder, the same skeleton taught for autoencoders, extended with skip
connections between matching Encoder/Decoder stages (not covered in the
course) so fine spatial detail lost at the Bottleneck can be recovered --
needed because our target is dense per-voxel classification, not Ch.10's
whole-image reconstruction.

Each encoder/decoder stage is a residual conv block (num_res_units=2): a
refinement of the plain conv-conv blocks in the original U-Net paper, adding
He et al.'s ResNet-style residual connections *within* each stage (on top of
the skip connections *between* stages). Downsampling/upsampling is done with
strided/transposed convolutions, matching Ch.4's convolution mechanics rather
than Ch.4's separate (parameter-free) pooling layers.

We use MONAI's UNet rather than a hand-rolled implementation: the conv
blocks, skip connections, and residual units are all standard, well-tested
components -- reimplementing them adds risk without teaching anything new.

Normalization deviates from Ch.3's default BatchNorm: with 3D patches, batch
size is small (memory-limited), where BatchNorm's running statistics are
unreliable. InstanceNorm normalizes each sample independently, avoiding that
problem -- same normalization idea taught in Ch.3, different granularity.
"""

from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_unet(dropout: float = 0.2) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # background, spleen -- see train.py for the softmax/one-hot convention
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
        dropout=dropout,
    )
