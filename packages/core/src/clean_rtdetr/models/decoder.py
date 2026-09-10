# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR decoder, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed component boundaries, explicit execution/output plan, and decoder-owned
prediction heads. See third_party/RT-DETR/README.md.
"""

import copy
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import cast

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.contracts import DecoderInput, DecoderOutput, DecoderPlan, EncodedFeatures
from clean_rtdetr.geometry import inverse_sigmoid
from clean_rtdetr.models._checkpoint import module_checkpoint_config
from clean_rtdetr.models.attention import MultiScaleDeformableAttention
from clean_rtdetr.models.interfaces import Decoder as DecoderInterface
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


class _ClampFloat16(nn.Module):
    """Clamp a residual to the finite float16 range used by RT-DETR v1."""

    def forward(self, value: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Return the elementwise-clamped residual."""
        return value.clamp(min=-65504, max=65504)


class DecoderLayer(nn.Module):
    """Self-attention, deformable cross-attention, and feed-forward decoder layer."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        cross_attention: MultiScaleDeformableAttention | None = None,
        residual_stabilizer: nn.Module | None = None,
    ) -> None:
        """Build decoder sublayers with injectable attention and FFN stabilization."""
        super().__init__()
        self.config = config
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.cross_attn = (
            MultiScaleDeformableAttention(
                config.hidden_dim,
                config.num_heads,
                config.num_levels,
                config.num_points,
            )
            if cross_attention is None
            else cross_attention
        )
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.linear1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.activation = functional.relu if config.activation == "relu" else functional.gelu
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.residual_stabilizer = (
            _ClampFloat16() if residual_stabilizer is None else residual_stabilizer
        )

    def forward(  # noqa: PLR0913, PLR0917
        self,
        target: Float[Tensor, "batch queries channels"],
        reference_points: Float[Tensor, "batch queries _levels coordinates"],
        memory: Float[Tensor, "batch sequence channels"],
        spatial_shapes: Int64[Tensor, "_levels 2"],
        query_position: Float[Tensor, "batch queries channels"],
        attention_mask: Bool[Tensor, "total_queries total_queries"] | None,
    ) -> Float[Tensor, "batch queries channels"]:
        """Update query content while preserving upstream residual/norm ordering.

        Each sublayer preserves ``[B, Q, C]`` and has its own residual and norm::

            target -> self-attention -> cross-attention -> feed-forward
                       other queries    encoder memory    channel mixing

        ``query_position`` enters the attention queries (and self-attention keys).
        ``reference_points`` locates cross-attention samples in the image pyramid.
        ``attention_mask`` restricts query-to-query interaction only.
        """
        target = self._self_attention(target, query_position, attention_mask)
        target = self._cross_attention(
            target, query_position, reference_points, memory, spatial_shapes
        )
        return self._feed_forward(target)

    def _self_attention(
        self,
        target: Float[Tensor, "batch queries channels"],
        query_position: Float[Tensor, "batch queries channels"],
        attention_mask: Bool[Tensor, "total_queries total_queries"] | None,
    ) -> Float[Tensor, "batch queries channels"]:
        """Attend among queries, then apply the first residual and normalization.

        With ``x = target`` and ``p = query_position``::

            a = MHA(query=x+p, key=x+p, value=x, mask=attention_mask)
            y = norm1(x + dropout1(a))

        Boolean mask entry ``[i, j] = True`` blocks query i from reading query j.
        Values contain content only, so positional features are not copied into x.
        """
        positioned = target + query_position
        attended, _ = self.self_attn(
            positioned,
            positioned,
            value=target,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.norm1(target + self.dropout1(attended))

    def _cross_attention(
        self,
        target: Float[Tensor, "batch queries channels"],
        query_position: Float[Tensor, "batch queries channels"],
        reference_points: Float[Tensor, "batch queries _levels coordinates"],
        memory: Float[Tensor, "batch sequence channels"],
        spatial_shapes: Int64[Tensor, "_levels 2"],
    ) -> Float[Tensor, "batch queries channels"]:
        """Sample encoder memory, then apply the second residual and normalization.

        ``cross_attn`` implements the multi-scale aggregation in Deformable DETR
        Eq. (3), https://arxiv.org/html/2010.04159v4. The positioned target
        predicts offsets and weights; references locate samples in each level.
        Flattened memory supplies their values, giving ``[B, Q, C]``::

            a = cross_attn(target + query_position, references, memory, shapes)
            y = norm2(target + dropout2(a))
        """
        attended = self.cross_attn(
            target + query_position,
            reference_points,
            memory,
            spatial_shapes,
        )
        return self.norm2(target + self.dropout2(attended))

    def _feed_forward(
        self, target: Float[Tensor, "batch queries channels"]
    ) -> Float[Tensor, "batch queries channels"]:
        """Transform channels and apply the injected residual stabilization."""
        feedforward = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(self.residual_stabilizer(target + self.dropout4(feedforward)))

    def checkpoint_config(self) -> dict[str, object]:
        """Record the configuration used to construct this layer's operations."""
        return {"config": asdict(self.config)}


class Decoder(DecoderInterface):
    """Run iterative RT-DETR refinement with decoder-owned layer prediction heads."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        layer: DecoderLayer | None = None,
        default_eval_plan: DecoderPlan | None = None,
    ) -> None:
        """Clone the layer prototype and retain an optional inference plan."""
        super().__init__()
        self.config = config
        self.default_eval_plan = DecoderPlan() if default_eval_plan is None else default_eval_plan
        if (
            self.default_eval_plan.num_layers is not None
            and not 1 <= self.default_eval_plan.num_layers <= config.num_layers
        ):
            msg = "Default eval plan must select a nonempty configured prefix"
            raise ValueError(msg)
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

    @property
    def hidden_dim(self) -> int:
        """Return the decoder hidden channel width."""
        return self.config.hidden_dim

    @property
    def num_classes(self) -> int:
        """Return the number of foreground classes predicted by the decoder."""
        return self.config.num_classes

    @property
    def num_levels(self) -> int:
        """Return the number of encoded feature levels consumed by the decoder."""
        return self.config.num_levels

    def forward(
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        *,
        plan: DecoderPlan | None = None,
    ) -> DecoderOutput:
        """Execute the requested layer prefix and collect predictions per the plan."""
        self._validate_inputs(queries, features)
        selected_plan = self._resolve_plan(plan)
        executed_layers = self._execution_depth(selected_plan)
        return self._decode(queries, features, executed_layers, selected_plan)

    def _resolve_plan(self, plan: DecoderPlan | None) -> DecoderPlan:
        """Apply the injected eval depth while preserving caller output choices."""
        if self.training:
            return DecoderPlan() if plan is None else plan
        if plan is None:
            return self.default_eval_plan
        if plan.num_layers is None and self.default_eval_plan.num_layers is not None:
            return DecoderPlan(
                num_layers=self.default_eval_plan.num_layers,
                collect_intermediate=plan.collect_intermediate,
            )
        return plan

    def _validate_inputs(self, queries: DecoderInput, features: EncodedFeatures) -> None:
        """Check query and memory dimensions against the configured decoder."""
        queries.validate_structure()
        features.validate_structure()
        self._validate_dimensions(queries, features)

    def _validate_dimensions(self, queries: DecoderInput, features: EncodedFeatures) -> None:
        """Check shared batch size, configured hidden width, and feature level count."""
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

    def _execution_depth(self, selected_plan: DecoderPlan) -> int:
        """Resolve and validate the requested layer prefix."""
        config = self.config
        executed_layers = (
            config.num_layers if selected_plan.num_layers is None else selected_plan.num_layers
        )
        if not 1 <= executed_layers <= config.num_layers:
            msg = "Decoder plan num_layers must select a nonempty configured prefix"
            raise ValueError(msg)
        return executed_layers

    def _decode(
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        executed_layers: int,
        selected_plan: DecoderPlan,
    ) -> DecoderOutput:
        """Share refinement arithmetic after boundary and execution-plan validation.

        Refinement uses ``refine(delta, ref) = sigmoid(delta + inverse_sigmoid(ref))``.
        During training, the previous layer's refined box has two routes::

            previous_refinement
                ├─ detach → sampling_reference ─┬→ position / attention
                │                               └→ refine(delta_i, sampling_reference)
                │                                      │
                │                                      └→ refinement_i → next iteration
                │
                └→ refine(delta_i, previous_refinement)
                       │
                       └→ loss boxes_i

        Here ``delta_i = box_head_i(content_i)`` is evaluated separately for each
        refinement, preserving the pinned implementation's head-call order.
        The two refinements have the same values but different gradient paths.
        Loss boxes can differentiate the preceding box head through
        ``previous_refinement``; the sampling route stops that gradient.
        The first layer uses its refinement directly as loss boxes. In eval,
        neither route detaches and the second box-head call is unnecessary.
        Query content keeps its gradient across layers in both modes.

        All requested layers execute even when only the last prediction is kept.
        Outputs stack as ``[retained_layers, B, Q, classes]`` and
        ``[retained_layers, B, Q, 4]``. The initial detach boundary belongs to
        ``QueryAssembler``; this internal method also serves fixed-shape export.
        """
        output = queries.content
        sampling_reference = queries.ref_logits.sigmoid()
        previous_refinement = sampling_reference
        layer_boxes = []
        layer_logits = []
        for index, layer in enumerate(self.layers[:executed_layers]):
            reference_input = sampling_reference.unsqueeze(2)
            query_position = self.query_pos_head(sampling_reference)
            output = layer(
                output,
                reference_input,
                features.memory,
                features.spatial_shapes,
                query_position,
                queries.attention_mask,
            )
            refined = (
                self.dec_bbox_head[index](output) + inverse_sigmoid(sampling_reference)
            ).sigmoid()
            if selected_plan.collect_intermediate or index == executed_layers - 1:
                layer_logits.append(self.dec_score_head[index](output))
                boxes = refined
                if self.training and index > 0:
                    boxes = (
                        self.dec_bbox_head[index](output) + inverse_sigmoid(previous_refinement)
                    ).sigmoid()
                layer_boxes.append(boxes)
            previous_refinement = refined
            sampling_reference = refined.detach() if self.training else refined

        return DecoderOutput(torch.stack(layer_logits), torch.stack(layer_boxes))

    def checkpoint_config(self) -> dict[str, object]:
        """Describe decoder dimensions, layers, and the injected eval plan."""
        return {
            "config": asdict(self.config),
            "default_eval_plan": asdict(self.default_eval_plan),
            "layers": [module_checkpoint_config(layer) for layer in self.layers],
        }
