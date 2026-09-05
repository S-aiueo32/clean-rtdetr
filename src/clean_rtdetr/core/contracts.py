"""Eager boundary checks; export lowering will use a separate tensor-only schema."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureSpec:
    channels: int
    stride: int

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.stride <= 0:
            raise ValueError("Feature channels and stride must be positive")


@dataclass(frozen=True)
class MultiScaleFeatures:
    levels: tuple[Tensor, ...]
    specs: tuple[FeatureSpec, ...]

    def validate(self) -> None:
        if not self.levels or len(self.levels) != len(self.specs):
            raise ValueError("Expected one FeatureSpec per nonempty feature level")
        previous_stride = 0
        first = self.levels[0]
        for level, spec in zip(self.levels, self.specs, strict=True):
            if level.ndim != 4 or level.shape[1] != spec.channels:
                raise ValueError("Features must have shape [B, channels, H, W] matching their spec")
            if min(level.shape) <= 0 or not level.is_floating_point():
                raise ValueError("Features must be nonempty floating-point tensors")
            if spec.stride <= previous_stride:
                raise ValueError("Feature levels must have strictly increasing strides")
            if (level.shape[0], level.dtype, level.device) != (
                first.shape[0],
                first.dtype,
                first.device,
            ):
                raise ValueError("Feature levels must share batch size, dtype, and device")
            previous_stride = spec.stride


@dataclass(frozen=True)
class EncodedFeatures:
    memory: Tensor  # [B, S, C], level order then row-major (W advances first)
    spatial_shapes: Tensor  # [N, 2], int64 (H, W)
    level_start_index: Tensor  # [N], int64 prefix sum

    def validate(self) -> None:
        if self.memory.ndim != 3 or min(self.memory.shape) <= 0:
            raise ValueError("Memory must have nonempty shape [B, S, C]")
        if not self.memory.is_floating_point():
            raise ValueError("Memory must be floating point")
        shapes = self.spatial_shapes
        if shapes.ndim != 2 or shapes.shape[1] != 2 or shapes.shape[0] == 0:
            raise ValueError("Spatial shapes must have nonempty shape [N, 2]")
        for tensor in (shapes, self.level_start_index):
            if tensor.dtype != torch.int64 or tensor.device != self.memory.device:
                raise ValueError("Shape metadata must be int64 on the memory device")
        if not bool((shapes > 0).all()):
            raise ValueError("Spatial dimensions must be positive")
        sizes = shapes.prod(dim=1)
        expected = sizes.cumsum(0) - sizes
        if int(sizes.sum()) != self.memory.shape[1]:
            raise ValueError("Memory sequence length must equal the sum of feature areas")
        if not torch.equal(self.level_start_index, expected):
            raise ValueError("Level offsets must equal the exclusive prefix sum of feature areas")


@dataclass(frozen=True)
class QuerySet:
    """RT-DETR query state, not a mandatory internal format for every research variant.

    Feature-initialized content is detached by QuerySelector. Reference logits
    retain gradients here; query assembly owns their eventual detach before decoding.
    """

    content: Tensor  # [B, Q, C]
    ref_logits: Tensor  # [B, Q, 4], normalized cxcywh before sigmoid
    source_indices: Tensor  # [B, Q], int64


@dataclass(frozen=True)
class SelectionOutput:
    queries: QuerySet
    encoder_logits: Tensor  # [B, Q, K], retains gradient for auxiliary loss
    encoder_boxes: Tensor  # [B, Q, 4], normalized cxcywh, retains gradient
