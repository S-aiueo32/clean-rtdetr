# Copyright 2026 Clean RT-DETR contributors
"""Specialize validated inference metadata without modifying the reference model."""

import copy
from typing import cast

import torch
from jaxtyping import Bool, Float, Float32, Int64
from torch import Tensor, nn

from clean_rtdetr.core.geometry import generate_anchors
from clean_rtdetr.models.attention import MultiScaleDeformableAttention
from clean_rtdetr.models.decoder import Decoder, DecoderLayer, DecoderPlan
from clean_rtdetr.models.rtdetr import RTDETR, BaselineRTDETR
from clean_rtdetr.models.selector import QuerySelector

DEFAULT_SHAPE = (1, 3, 640, 640)


class _FixedAttention(MultiScaleDeformableAttention):
    """Reuse attention arithmetic with metadata checked before graph capture."""

    def __init__(
        self, attention: MultiScaleDeformableAttention, shapes: tuple[tuple[int, int], ...]
    ) -> None:
        nn.Module.__init__(self)
        self.attention = attention
        self.shapes = shapes

    def forward(
        self,
        query: Float[Tensor, "batch queries channels"],
        reference_points: Float[Tensor, "batch queries levels coordinates"],
        value: Float[Tensor, "batch sequence channels"],
        spatial_shapes: Int64[Tensor, "feature_levels 2"],
    ) -> Float[Tensor, "batch queries channels"]:
        """Evaluate the shared sampling kernel using fixed split/reshape sizes."""
        return self.attention._attend(  # noqa: SLF001 - explicit internal lowering boundary
            query, reference_points, value, spatial_shapes, self.shapes
        )


class FixedShapeRTDETR(nn.Module):
    """Own an eval CPU FP32 copy returning schema-v1 ``(logits, boxes)`` tensors.

    The deployment profile is batch one at 640x640. Smaller fixed multiples of
    32 are accepted for component regression tests, not advertised AP profiles.
    All configured decoder layers execute; boxes are normalized cxcywh.
    """

    anchors: Float32[Tensor, "1 sequence 4"]
    valid: Bool[Tensor, "1 sequence 1"]

    def __init__(
        self, model: BaselineRTDETR, input_shape: tuple[int, int, int, int] = DEFAULT_SHAPE
    ) -> None:
        """Validate the eager path, freeze metadata, and specialize the copied attention."""
        super().__init__()
        if (
            len(input_shape) != 4  # noqa: PLR2004 - NCHW
            or input_shape[:2] != (1, 3)
            or any(type(size) is not int or size <= 0 for size in input_shape)
            or any(size % 32 for size in input_shape[2:])
        ):
            msg = "Expected fixed [1, 3, H, W] with positive integer multiples of 32"
            raise ValueError(msg)
        if (
            type(model) is not RTDETR
            or type(model.selector) is not QuerySelector
            or type(model.decoder) is not Decoder
            or any(
                type(layer) is not DecoderLayer
                or type(layer.cross_attn) is not MultiScaleDeformableAttention
                for layer in model.decoder.layers
            )
        ):
            msg = "Fixed-shape lowering requires the built-in RTDETR selector and decoder types"
            raise ValueError(msg)
        self.input_shape = input_shape
        self.model = copy.deepcopy(model).cpu().float().eval()
        self.plan = DecoderPlan(collect_intermediate=False)
        with torch.no_grad():
            images = torch.zeros(input_shape)
            features = self.model.backbone(images)
            encoded = self.model.projector(self.model.encoder(features).features)
            selected = self.model.selector(encoded)
            self.model.decoder(self.model.assembler(selected.queries), encoded, plan=self.plan)
            shapes = tuple((int(h), int(w)) for h, w in encoded.spatial_shapes.tolist())
            anchors, valid = generate_anchors(shapes, eps=self.model.selector.config.anchor_eps)
        self.register_buffer("anchors", anchors)
        self.register_buffer("valid", valid)
        for layer in self.model.decoder.layers:
            decoder_layer = cast("DecoderLayer", layer)
            decoder_layer.cross_attn = _FixedAttention(decoder_layer.cross_attn, shapes)
        self.eval()

    def forward(
        self, images: Float32[Tensor, "1 3 height width"]
    ) -> tuple[Float32[Tensor, "1 queries classes"], Float32[Tensor, "1 queries 4"]]:
        """Reject inputs outside the fixed profile and return final raw predictions."""
        if images.shape != self.input_shape:
            msg = f"Expected fixed input shape {self.input_shape}"
            raise ValueError(msg)
        if images.dtype != torch.float32 or images.device.type != "cpu":
            msg = "Fixed-shape export expects CPU float32 input"
            raise ValueError(msg)
        if self.training or any(module.training for module in self.model.modules()):
            msg = "Fixed-shape export requires eval mode"
            raise ValueError(msg)
        features = self.model.backbone(images)
        encoded = self.model.projector(self.model.encoder(features).features)
        selected = self.model.selector._select(  # noqa: SLF001 - prevalidated metadata
            encoded.memory, self.anchors, self.valid
        )
        queries = self.model.assembler(selected.queries)
        decoded = self.model.decoder._decode(  # noqa: SLF001 - fixed inference plan
            queries, encoded, self.model.decoder.config.num_layers, self.plan
        )
        return decoded.logits[-1], decoded.boxes[-1]
