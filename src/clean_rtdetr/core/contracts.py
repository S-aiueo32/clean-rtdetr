# Copyright 2026 Clean RT-DETR contributors
"""Eager boundary checks; export lowering will use a separate tensor-only schema."""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int64
from torch import Tensor

FEATURE_RANK = 4
MEMORY_RANK = 3
SPATIAL_SHAPE_RANK = 2
SPATIAL_DIMENSIONS = 2


@dataclass(frozen=True)
class FeatureSpec:
    """Describe the channels and input-relative stride of one feature level."""

    channels: int
    stride: int

    def __post_init__(self) -> None:
        """Reject nonpositive channel counts and strides."""
        if self.channels <= 0 or self.stride <= 0:
            msg = "Feature channels and stride must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class MultiScaleFeatures:
    """Pair NCHW feature levels with their ordered construction specifications."""

    levels: tuple[Float[Tensor, "batch _channels _height _width"], ...]
    specs: tuple[FeatureSpec, ...]

    def validate(self) -> None:
        """Check feature layout, spec ordering, and shared batch/dtype/device."""
        if not self.levels or len(self.levels) != len(self.specs):
            msg = "Expected one FeatureSpec per nonempty feature level"
            raise ValueError(msg)
        previous_stride = 0
        first = self.levels[0]
        for level, spec in zip(self.levels, self.specs, strict=True):
            if level.ndim != FEATURE_RANK or level.shape[1] != spec.channels:
                msg = "Features must have shape [B, channels, H, W] matching their spec"
                raise ValueError(msg)
            if min(level.shape) <= 0 or not level.is_floating_point():
                msg = "Features must be nonempty floating-point tensors"
                raise ValueError(msg)
            if spec.stride <= previous_stride:
                msg = "Feature levels must have strictly increasing strides"
                raise ValueError(msg)
            if (level.shape[0], level.dtype, level.device) != (
                first.shape[0],
                first.dtype,
                first.device,
            ):
                msg = "Feature levels must share batch size, dtype, and device"
                raise ValueError(msg)
            previous_stride = spec.stride


@dataclass(frozen=True)
class EncodedFeatures:
    """Store projected memory and the spatial metadata needed by the decoder."""

    memory: Float[Tensor, "batch sequence channels"]  # level order, then W advances first
    spatial_shapes: Int64[Tensor, "levels 2"]  # (H, W)
    level_start_index: Int64[Tensor, " levels"]  # leading space avoids Ruff F821; prefix sum

    def validate(self) -> None:
        """Check memory layout and the exact level area/offset correspondence."""
        if self.memory.ndim != MEMORY_RANK or min(self.memory.shape) <= 0:
            msg = "Memory must have nonempty shape [B, S, C]"
            raise ValueError(msg)
        if not self.memory.is_floating_point():
            msg = "Memory must be floating point"
            raise ValueError(msg)
        shapes = self.spatial_shapes
        if (
            shapes.ndim != SPATIAL_SHAPE_RANK
            or shapes.shape[1] != SPATIAL_DIMENSIONS
            or shapes.shape[0] == 0
        ):
            msg = "Spatial shapes must have nonempty shape [N, 2]"
            raise ValueError(msg)
        for tensor in (shapes, self.level_start_index):
            if tensor.dtype != torch.int64 or tensor.device != self.memory.device:
                msg = "Shape metadata must be int64 on the memory device"
                raise ValueError(msg)
        if not bool((shapes > 0).all()):
            msg = "Spatial dimensions must be positive"
            raise ValueError(msg)
        sizes = shapes.prod(dim=1)
        expected = sizes.cumsum(0) - sizes
        if int(sizes.sum()) != self.memory.shape[1]:
            msg = "Memory sequence length must equal the sum of feature areas"
            raise ValueError(msg)
        if not torch.equal(self.level_start_index, expected):
            msg = "Level offsets must equal the exclusive prefix sum of feature areas"
            raise ValueError(msg)


@dataclass(frozen=True)
class QuerySet:
    """RT-DETR query state, not a mandatory internal format for every research variant.

    Feature-initialized content is detached by QuerySelector. Reference logits
    retain gradients here; query assembly owns their eventual detach before decoding.
    """

    content: Float[Tensor, "batch queries channels"]
    ref_logits: Float[Tensor, "batch queries 4"]  # normalized cxcywh before sigmoid
    source_indices: Int64[Tensor, "batch queries"]


@dataclass(frozen=True)
class SelectionOutput:
    """Keep initialized queries and differentiable encoder auxiliary predictions."""

    queries: QuerySet
    encoder_logits: Float[Tensor, "batch queries classes"]  # retains auxiliary loss gradient
    encoder_boxes: Float[Tensor, "batch queries 4"]  # normalized cxcywh, retains gradient
