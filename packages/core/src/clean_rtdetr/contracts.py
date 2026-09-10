# Copyright 2026 Clean RT-DETR contributors
"""Typed tensor contracts with explicit boundary and internal structure checks."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor

FEATURE_RANK = 4
MEMORY_RANK = 3
QUERY_RANK = 3
DECODER_OUTPUT_RANK = 4
BOX_COORDINATES = 4
SPATIAL_SHAPE_RANK = 2
SPATIAL_DIMENSIONS = 2


@dataclass(frozen=True)
class DecoderPlan:
    """Choose the executed prefix and whether to retain intermediate predictions."""

    num_layers: int | None = None
    collect_intermediate: bool = True


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
        """Check annotated layouts as well as feature metadata and placement."""
        for level in self.levels:
            if level.ndim != FEATURE_RANK:
                msg = "Features must have shape [B, channels, H, W] matching their spec"
                raise ValueError(msg)
            if not level.is_floating_point():
                msg = "Features must be nonempty floating-point tensors"
                raise ValueError(msg)
            if level.shape[0] != self.levels[0].shape[0]:
                msg = "Feature levels must share batch size, dtype, and device"
                raise ValueError(msg)
        self.validate_structure()

    def validate_structure(self) -> None:
        """Check metadata and placement while trusting annotated tensor layouts."""
        if not self.levels or len(self.levels) != len(self.specs):
            msg = "Expected one FeatureSpec per nonempty feature level"
            raise ValueError(msg)
        previous_stride = 0
        for level, spec in zip(self.levels, self.specs, strict=True):
            self._validate_level_structure(level, spec, self.levels[0], previous_stride)
            previous_stride = spec.stride

    @staticmethod
    def _validate_level_structure(
        level: Shaped[Tensor, "..."],
        spec: FeatureSpec,
        first: Shaped[Tensor, "..."],
        previous_stride: int,
    ) -> None:
        """Check dimensions tied to specs, nonempty axes, and shared placement."""
        if level.shape[1] != spec.channels:
            msg = "Feature channels must match their spec"
            raise ValueError(msg)
        if min(level.shape) <= 0:
            msg = "Features must be nonempty tensors"
            raise ValueError(msg)
        if spec.stride <= previous_stride:
            msg = "Feature levels must have strictly increasing strides"
            raise ValueError(msg)
        if (level.dtype, level.device) != (first.dtype, first.device):
            msg = "Feature levels must share batch size, dtype, and device"
            raise ValueError(msg)


@dataclass(frozen=True)
class HybridEncoderOutput:
    """Expose fused features and optionally the encoded level before cross-scale fusion.

    ``after_aifi`` retains its autograd graph and contains only the stride-32
    level processed by AIFI. It is absent unless explicitly requested.
    """

    features: MultiScaleFeatures
    after_aifi: MultiScaleFeatures | None


@dataclass(frozen=True)
class EncodedFeatures:
    """Store projected memory and the spatial metadata needed by the decoder."""

    memory: Float[Tensor, "batch sequence channels"]  # level order, then W advances first
    spatial_shapes: Int64[Tensor, "levels 2"]  # (H, W)
    level_start_index: Int64[Tensor, " levels"]  # leading space avoids Ruff F821; prefix sum

    def validate(self) -> None:
        """Check annotated layouts as well as spatial values and placement."""
        if self.memory.ndim != MEMORY_RANK:
            msg = "Memory must have nonempty shape [B, S, C]"
            raise ValueError(msg)
        if not self.memory.is_floating_point():
            msg = "Memory must be floating point"
            raise ValueError(msg)
        if (
            self.spatial_shapes.ndim != SPATIAL_SHAPE_RANK
            or self.spatial_shapes.shape[1] != SPATIAL_DIMENSIONS
        ):
            msg = "Spatial shapes must have nonempty shape [N, 2]"
            raise ValueError(msg)
        for tensor in (self.spatial_shapes, self.level_start_index):
            if tensor.dtype != torch.int64:
                msg = "Shape metadata must be int64 on the memory device"
                raise ValueError(msg)
        self.validate_structure()

    def validate_structure(self) -> None:
        """Check nonempty memory, spatial values, offsets, and metadata placement."""
        if min(self.memory.shape) <= 0:
            msg = "Memory must have nonempty shape [B, S, C]"
            raise ValueError(msg)
        if self.spatial_shapes.shape[0] == 0:
            msg = "Spatial shapes must have nonempty shape [N, 2]"
            raise ValueError(msg)
        if any(
            tensor.device != self.memory.device
            for tensor in (self.spatial_shapes, self.level_start_index)
        ):
            msg = "Shape metadata must be on the memory device"
            raise ValueError(msg)
        if not bool((self.spatial_shapes > 0).all()):
            msg = "Spatial dimensions must be positive"
            raise ValueError(msg)
        self._validate_area_and_offsets()

    def _validate_area_and_offsets(self) -> None:
        """Check sequence length and exclusive prefix sums from level areas."""
        sizes = self.spatial_shapes.prod(dim=1)
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


@dataclass(frozen=True)
class DenoisingQueries:
    """Prepared training queries; their construction belongs to the training package."""

    content: Float[Tensor, "batch denoising channels"]
    ref_logits: Float[Tensor, "batch denoising 4"]
    attention_mask: Bool[Tensor, "total_queries total_queries"]  # True blocks attention


@dataclass(frozen=True)
class DecoderInput:
    """Combined matching/DN queries with references detached at this boundary."""

    content: Float[Tensor, "batch total_queries channels"]
    ref_logits: Float[Tensor, "batch total_queries 4"]
    attention_mask: Bool[Tensor, "total_queries total_queries"] | None
    matching_queries: int

    def validate(self) -> None:
        """Check annotated query layouts as well as placement and detach rules."""
        self._validate_query_shapes()
        if not self.content.is_floating_point() or not self.ref_logits.is_floating_point():
            msg = "Query content and references must share floating dtype and device"
            raise ValueError(msg)
        self._validate_attention_mask(self.content.shape[1])
        self.validate_structure()

    def validate_structure(self) -> None:
        """Check nonempty queries, placement, matching count, and detached references."""
        if min(self.content.shape) <= 0:
            msg = "Query content must have nonempty shape [B, Q, C]"
            raise ValueError(msg)
        if (self.content.dtype, self.content.device) != (
            self.ref_logits.dtype,
            self.ref_logits.device,
        ):
            msg = "Query content and references must share floating dtype and device"
            raise ValueError(msg)
        if self.ref_logits.requires_grad:
            msg = "Decoder reference logits must be detached during query assembly"
            raise ValueError(msg)
        self._validate_matching_query_count(self.content.shape[1])
        if self.attention_mask is not None and self.attention_mask.device != self.content.device:
            msg = "Attention mask must be on the query device"
            raise ValueError(msg)

    def _validate_query_shapes(self) -> None:
        """Validate the content layout and matching reference dimensions."""
        if self.content.ndim != QUERY_RANK or min(self.content.shape) <= 0:
            msg = "Query content must have nonempty shape [B, Q, C]"
            raise ValueError(msg)
        batch, total_queries, _ = self.content.shape
        if self.ref_logits.shape != (batch, total_queries, BOX_COORDINATES):
            msg = "Reference logits must have shape [B, Q, 4]"
            raise ValueError(msg)

    def _validate_matching_query_count(self, total_queries: int) -> None:
        """Validate the size of the matching suffix within the assembled query sequence."""
        if not 0 < self.matching_queries <= total_queries:
            msg = "Matching query count must be within the assembled query sequence"
            raise ValueError(msg)

    def _validate_attention_mask(self, total_queries: int) -> None:
        """Validate an optional boolean self-attention mask on the query device."""
        if self.attention_mask is not None and (
            self.attention_mask.shape != (total_queries, total_queries)
            or self.attention_mask.dtype != torch.bool
            or self.attention_mask.device != self.content.device
        ):
            msg = "Attention mask must be bool [Q, Q] on the query device"
            raise ValueError(msg)


@dataclass(frozen=True)
class DetectionOutput:
    """Class logits and normalized cxcywh boxes from one decoder layer."""

    logits: Float[Tensor, "batch queries classes"]
    boxes: Float[Tensor, "batch queries 4"]


@dataclass(frozen=True)
class DecoderOutput:
    """Predictions in execution order for the layers selected by the explicit plan."""

    logits: Float[Tensor, "layers batch queries classes"]
    boxes: Float[Tensor, "layers batch queries 4"]

    @property
    def final(self) -> DetectionOutput:
        """Return the final executed decoder layer."""
        return DetectionOutput(self.logits[-1], self.boxes[-1])

    def validate(self) -> None:
        """Check stacked layer outputs share their leading dimensions."""
        self._validate_output_shapes()
        self._validate_output_tensors()

    def _validate_output_shapes(self) -> None:
        """Validate ranks, box coordinates, and shared layer/batch/query dimensions."""
        if self.logits.ndim != DECODER_OUTPUT_RANK or self.boxes.ndim != DECODER_OUTPUT_RANK:
            msg = "Decoder outputs must have shape [L, B, Q, K/4]"
            raise ValueError(msg)

        if (
            self.boxes.shape[-1] != BOX_COORDINATES
            or self.logits.shape[:QUERY_RANK] != self.boxes.shape[:QUERY_RANK]
        ):
            msg = "Decoder logits and boxes must share [L, B, Q] dimensions"
            raise ValueError(msg)

    def _validate_output_tensors(self) -> None:
        """Validate nonempty floating logits and shared floating dtype/device."""
        if min(self.logits.shape) <= 0 or not self.logits.is_floating_point():
            msg = "Decoder logits must be nonempty floating-point tensors"
            raise ValueError(msg)

        if (
            not self.boxes.is_floating_point()
            or self.logits.dtype != self.boxes.dtype
            or self.logits.device != self.boxes.device
        ):
            msg = "Decoder logits and boxes must share floating dtype and device"
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelOutput:
    """Return image-to-prediction results with the same fields in train and eval.

    Encoder auxiliary predictions retain gradients. AIFI features are present
    only when requested. Every supplied query remains in execution order;
    assigning predictions to supervision groups belongs to the caller.
    """

    decoder: DecoderOutput
    encoder_aux: DetectionOutput
    after_aifi: MultiScaleFeatures | None

    @property
    def final(self) -> DetectionOutput:
        """Return logits and normalized cxcywh boxes from the last executed layer."""
        return self.decoder.final
