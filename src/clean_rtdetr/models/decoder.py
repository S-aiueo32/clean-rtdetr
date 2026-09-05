# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR decoder, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed component boundaries, explicit execution/output plan, and decoder-owned
prediction heads. See third_party/RT-DETR/README.md.
"""

import copy
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.core.contracts import DecoderInput, DecoderOutput, EncodedFeatures
from clean_rtdetr.core.geometry import inverse_sigmoid
from clean_rtdetr.models.attention import MultiScaleDeformableAttention
from clean_rtdetr.models.selector import BoxHead


@dataclass(frozen=True)
class DecoderConfig:
    """Define baseline decoder dimensions and attention structure."""

    hidden_dim: int = 256
    num_classes: int = 80
    num_layers: int = 6
    num_heads: int = 8
    num_levels: int = 3
    num_points: int = 4
    feedforward_dim: int = 1024
    dropout: float = 0.0
    activation: str = "relu"

    def __post_init__(self) -> None:
        """Reject dimensions and options that cannot form an upstream decoder."""
        dimensions = (
            self.hidden_dim,
            self.num_classes,
            self.num_layers,
            self.num_heads,
            self.num_levels,
            self.num_points,
            self.feedforward_dim,
        )
        if min(dimensions) <= 0:
            msg = "Decoder dimensions must be positive"
            raise ValueError(msg)
        if self.hidden_dim % self.num_heads:
            msg = "Decoder hidden dimension must be divisible by num_heads"
            raise ValueError(msg)
        if not 0 <= self.dropout < 1:
            msg = "Decoder dropout must be in [0, 1)"
            raise ValueError(msg)
        if self.activation not in {"relu", "gelu"}:
            msg = "Decoder activation must be relu or gelu"
            raise ValueError(msg)


@dataclass(frozen=True)
class DecoderPlan:
    """Choose the executed prefix and whether to retain intermediate predictions."""

    num_layers: int | None = None
    collect_intermediate: bool = True


class _MLP(nn.Module):
    """Linear stack with activation after every layer except the last."""

    def __init__(self, dimensions: tuple[int, ...], activation: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim) for input_dim, output_dim in pairwise(dimensions)
        )
        self.activation = activation

    def forward(
        self, values: Float[Tensor, "*batch channels"]
    ) -> Float[Tensor, "*batch channels_out"]:
        """Apply the configured linear layers."""
        for layer in self.layers[:-1]:
            values = self.activation(layer(values))
        return self.layers[-1](values)


class DecoderLayer(nn.Module):
    """Self-attention, deformable cross-attention, and feed-forward decoder layer."""

    def __init__(self, config: DecoderConfig) -> None:
        """Build the three upstream decoder sublayers."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.cross_attn = MultiScaleDeformableAttention(
            config.hidden_dim,
            config.num_heads,
            config.num_levels,
            config.num_points,
        )
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.linear1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.activation = functional.relu if config.activation == "relu" else functional.gelu
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.hidden_dim)

    def forward(  # noqa: PLR0913, PLR0917
        self,
        target: Float[Tensor, "batch queries channels"],
        reference_points: Float[Tensor, "batch queries _levels coordinates"],
        memory: Float[Tensor, "batch sequence channels"],
        spatial_shapes: Int64[Tensor, "_levels 2"],
        query_position: Float[Tensor, "batch queries channels"],
        attention_mask: Bool[Tensor, "total_queries total_queries"] | None,
    ) -> Float[Tensor, "batch queries channels"]:
        """Update query content while preserving upstream residual/norm ordering."""
        positioned = target + query_position
        attended, _ = self.self_attn(
            positioned,
            positioned,
            value=target,
            attn_mask=attention_mask,
            need_weights=False,
        )
        target = self.norm1(target + self.dropout1(attended))
        attended = self.cross_attn(
            target + query_position,
            reference_points,
            memory,
            spatial_shapes,
        )
        target = self.norm2(target + self.dropout2(attended))
        feedforward = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3((target + self.dropout4(feedforward)).clamp(min=-65504, max=65504))


class Decoder(nn.Module):
    """Run iterative RT-DETR refinement with decoder-owned layer prediction heads."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        layer: DecoderLayer | None = None,
    ) -> None:
        """Clone the layer prototype and construct independent prediction heads."""
        super().__init__()
        self.config = config
        prototype = DecoderLayer(config) if layer is None else layer
        self.layers = nn.ModuleList(copy.deepcopy(prototype) for _ in range(config.num_layers))
        self.query_pos_head = _MLP(
            (4, 2 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.dec_score_head = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.num_classes) for _ in range(config.num_layers)
        )
        self.dec_bbox_head = nn.ModuleList(
            BoxHead(config.hidden_dim) for _ in range(config.num_layers)
        )
        classification_bias = -math.log(99.0)
        for score_head in self.dec_score_head:
            nn.init.constant_(cast("nn.Linear", score_head).bias, classification_bias)
        for position_layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(cast("nn.Linear", position_layer).weight)

    def forward(
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        *,
        plan: DecoderPlan | None = None,
    ) -> DecoderOutput:
        """Execute the requested layer prefix and collect predictions per the plan."""
        queries.validate()
        features.validate()
        config = self.config
        if queries.content.shape[0] != features.memory.shape[0]:
            msg = "Queries and memory must share batch size"
            raise ValueError(msg)
        if (
            queries.content.shape[2] != config.hidden_dim
            or features.memory.shape[2] != config.hidden_dim
        ):
            msg = "Queries and memory must match the decoder hidden dimension"
            raise ValueError(msg)
        if features.spatial_shapes.shape[0] != config.num_levels:
            msg = "Encoded feature levels must match the decoder configuration"
            raise ValueError(msg)
        selected_plan = DecoderPlan() if plan is None else plan
        executed_layers = (
            config.num_layers if selected_plan.num_layers is None else selected_plan.num_layers
        )
        if not 1 <= executed_layers <= config.num_layers:
            msg = "Decoder plan num_layers must select a nonempty configured prefix"
            raise ValueError(msg)
        return self._decode(queries, features, executed_layers, selected_plan)

    def _decode(
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        executed_layers: int,
        selected_plan: DecoderPlan,
    ) -> DecoderOutput:
        """Share refinement arithmetic after boundary and execution-plan validation."""
        output = queries.content
        detached_reference = queries.ref_logits.sigmoid()
        previous_reference = detached_reference
        layer_boxes = []
        layer_logits = []
        for index, layer in enumerate(self.layers[:executed_layers]):
            reference_input = detached_reference.unsqueeze(2)
            query_position = self.query_pos_head(detached_reference)
            output = layer(
                output,
                reference_input,
                features.memory,
                features.spatial_shapes,
                query_position,
                queries.attention_mask,
            )
            refined = (
                self.dec_bbox_head[index](output) + inverse_sigmoid(detached_reference)
            ).sigmoid()
            if selected_plan.collect_intermediate or index == executed_layers - 1:
                layer_logits.append(self.dec_score_head[index](output))
                boxes = refined
                if self.training and index > 0:
                    boxes = (
                        self.dec_bbox_head[index](output) + inverse_sigmoid(previous_reference)
                    ).sigmoid()
                layer_boxes.append(boxes)
            previous_reference = refined
            detached_reference = refined.detach() if self.training else refined

        result = DecoderOutput(torch.stack(layer_logits), torch.stack(layer_boxes))
        result.validate()
        return result
