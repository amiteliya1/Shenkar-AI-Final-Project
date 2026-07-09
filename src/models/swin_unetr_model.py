"""Swin UNETR: Swin Transformer encoder + CNN decoder for 3D segmentation.

Course framing extends Ch.6 (Attention/Transformers) beyond its word-token
setting: the same `Q=X*W_Q, K=X*W_K, V=X*W_V`,
`Attention=Softmax(QK^T/sqrt(d_k))*V` formula is computed here over flattened
3D image patches instead of word tokens, restricted to local, non-overlapping
windows (reducing Ch.6's stated O(n^2) self-attention cost to linear in patch
count), with a relative positional bias replacing Ch.6's additive Positional
Encoding -- both solve the same "attention has no sense of order by itself"
problem Ch.6 describes.

Decoder side mirrors the baseline U-Net: Ch.10's Encoder -> Bottleneck ->
Decoder skeleton with skip connections (see unet_baseline.py), so the two
models differ only in *how* the Encoder computes features (windowed
self-attention here vs. convolution there) -- loss, metric, and data
pipeline (src/data/transforms.py, src/data/dataset.py) are identical for both,
so the comparison isolates the encoder choice.

Uses MONAI's SwinUNETR rather than a hand-rolled implementation, for the same
reason as the baseline: the windowing/shifting/patch-merging logic is
intricate, well-tested infrastructure, not something worth reimplementing.

GPU memory (NVIDIA L4, 24GB): Swin UNETR's attention layers are considerably
more memory-hungry per patch than the baseline U-Net's convolutions, so two
deliberate choices keep it feasible at the *same* 96^3 patch size the
baseline uses (patch size must match for a fair comparison, per the project
plan -- enforced by both models sharing src/data/transforms.py's PATCH_SIZE
and train.py's sliding_window_inference(roi_size=PATCH_SIZE, ...), not by
anything in this file):
  - `feature_size=24` -- MONAI's own default, and the lighter of the two
    configs used in the original Swin UNETR paper (vs. 48 for BraTS).
  - `use_checkpoint=True` -- activation/gradient checkpointing, trading extra
    compute for much lower activation memory.
Batch size is the other lever, but that's a per-run training hyperparameter
(configs/*.yaml's `batch_size`), not an architectural one, so it isn't set here.

Note: the installed MONAI release no longer accepts an `img_size` argument
here -- newer SwinUNETR versions are fully input-size-agnostic at
construction time (relative position bias is window-size-, not image-size-,
based), so it was removed entirely rather than reintroduced. The 96^3 patch
size is still enforced, just from the data/inference side instead.
"""

from monai.networks.nets import SwinUNETR


def build_swin_unetr(dropout: float = 0.2) -> SwinUNETR:
    return SwinUNETR(
        in_channels=1,
        out_channels=2,  # background, spleen -- matches unet_baseline.py's convention
        feature_size=24,
        norm_name="instance",  # same InstanceNorm rationale as the baseline (small batch sizes)
        drop_rate=dropout,
        use_checkpoint=True,
        spatial_dims=3,
    )
