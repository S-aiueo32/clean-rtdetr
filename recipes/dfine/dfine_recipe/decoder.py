# Copyright 2026 Clean RT-DETR contributors
"""Recipe-local FDR decoder built on the public RT-DETR decoder layer."""

import copy
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.contracts import (
    DecoderInput,
    DecoderOutput,
    DecoderPlan,
    DetectionOutput,
    EncodedFeatures,
)
from clean_rtdetr.geometry import inverse_sigmoid
from clean_rtdetr.models.decoder import DecoderConfig, DecoderLayer
from clean_rtdetr.models.interfaces import Decoder as DecoderInterface
from clean_rtdetr.models.selector import BoxHead

from .distribution import distance_to_boxes, weighting_function


@dataclass(frozen=True)
class DistributionOutput(DecoderOutput):
    """Decoder outputs with FDR corner distributions and fixed references."""

    corners: Float[Tensor, "layers batch queries 4 bins"]
    references: Float[Tensor, "batch queries 4"]
    reg_scale: float
    up: float
    pre: DetectionOutput

    def validate(self) -> None:
        """Validate base predictions and all FDR tensors."""
        super().validate()
        layers, batch, queries, _ = self.logits.shape
        if self.corners.shape[:3] != (layers, batch, queries) or self.corners.shape[-2] != 4:  # noqa: PLR2004
            msg = "FDR corners must have shape [L, B, Q, 4, bins]"
            raise ValueError(msg)
        if self.references.shape != (batch, queries, 4):
            msg = "FDR references must have shape [B, Q, 4]"
            raise ValueError(msg)
        if (
            self.corners.ndim != 5  # noqa: PLR2004
            or self.corners.dtype != self.logits.dtype
            or self.corners.device != self.logits.device
            or not torch.isfinite(self.corners).all()
            or self.references.dtype != self.logits.dtype
            or self.references.device != self.logits.device
            or not torch.isfinite(self.references).all()
        ):
            msg = "FDR tensors must be finite and share logits dtype and device"
            raise ValueError(msg)
        if (
            not math.isfinite(self.reg_scale)
            or self.reg_scale <= 0
            or not math.isfinite(self.up)
            or self.up <= 0
        ):
            msg = "FDR distribution parameters must be finite and positive"
            raise ValueError(msg)
        if (
            self.pre.logits.ndim != 3  # noqa: PLR2004
            or self.pre.boxes.ndim != 3  # noqa: PLR2004
            or self.pre.logits.shape[:2] != (batch, queries)
            or self.pre.boxes.shape != (batch, queries, 4)
            or self.pre.logits.shape[-1] != self.logits.shape[-1]
            or self.pre.boxes.dtype != self.logits.dtype
            or self.pre.logits.dtype != self.logits.dtype
            or self.pre.boxes.device != self.logits.device
            or not torch.isfinite(self.pre.logits).all()
            or not torch.isfinite(self.pre.boxes).all()
            or self.pre.logits.device != self.logits.device
        ):
            msg = "FDR pre predictions must share batch, query, dtype, and device"
            raise ValueError(msg)


class _MLP(nn.Module):
    """Small linear stack used for query positions and distributions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            msg = "MLP must contain at least one layer"
            raise ValueError(msg)
        dimensions = [input_dim, *([hidden_dim] * (layers - 1)), output_dim]
        self.layers = nn.ModuleList(nn.Linear(left, right) for left, right in pairwise(dimensions))

    def forward(self, values: Float[Tensor, "*batch channels"]) -> Tensor:
        """Apply ReLU between linear layers."""
        for layer in self.layers[:-1]:
            values = functional.relu(layer(values))
        return self.layers[-1](values)


class _LQE(nn.Module):
    """Official D-FINE local quality estimator for distribution logits."""

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.reg_conf = _MLP(4 * 5, 64, 1, 2)
        final = cast("nn.Linear", self.reg_conf.layers[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        scores: Float[Tensor, "batch queries classes"],
        corners: Float[Tensor, "batch queries 4 bins"],
    ) -> Tensor:
        """Add learned quality from the top four bin probabilities per edge."""
        probability = functional.softmax(corners, dim=-1)
        top, _ = probability.topk(4, dim=-1)
        statistics = torch.cat((top, top.mean(dim=-1, keepdim=True)), dim=-1)
        return scores + self.reg_conf(statistics.reshape(*statistics.shape[:2], -1))


class FDRDecoder(DecoderInterface):
    """Hybrid RT-DETR decoder with D-FINE fine-grained distribution refinement."""

    def __init__(  # noqa: PLR0913
        self,
        config: DecoderConfig,
        *,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        up: float = 0.5,
        use_lqe: bool = True,
        default_eval_plan: DecoderPlan | None = None,
    ) -> None:
        """Build independent public RT-DETR layers and recipe-local FDR heads."""
        super().__init__()
        if not math.isfinite(up) or up <= 0:
            msg = "up must be positive and finite"
            raise ValueError(msg)
        self.config = config
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.up = up
        self.use_lqe = use_lqe
        weighting_function(reg_max, up, reg_scale, torch.zeros(1))
        self.default_eval_plan = DecoderPlan() if default_eval_plan is None else default_eval_plan
        if (
            self.default_eval_plan.num_layers is not None
            and not 1 <= self.default_eval_plan.num_layers <= config.num_layers
        ):
            msg = "Default eval plan must select a configured prefix"
            raise ValueError(msg)
        prototype = DecoderLayer(config)
        self.layers = nn.ModuleList(copy.deepcopy(prototype) for _ in range(config.num_layers))
        self.query_pos_head = _MLP(4, 2 * config.hidden_dim, config.hidden_dim, 2)
        self.pre_bbox_head = BoxHead(config.hidden_dim)
        self.score_heads = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.num_classes) for _ in range(config.num_layers)
        )
        self.distribution_heads = nn.ModuleList(
            _MLP(config.hidden_dim, config.hidden_dim, 4 * (reg_max + 1), 3)
            for _ in range(config.num_layers)
        )
        for head in self.distribution_heads:
            final = cast("nn.Linear", cast("_MLP", head).layers[-1])
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.lqe = nn.ModuleList(
            _LQE(reg_max) if use_lqe else nn.Identity() for _ in range(config.num_layers)
        )
        for head in self.score_heads:
            nn.init.constant_(cast("nn.Linear", head).bias, -math.log(99.0))

    @property
    def hidden_dim(self) -> int:
        """Return decoder channel width."""
        return self.config.hidden_dim

    @property
    def num_classes(self) -> int:
        """Return foreground class count."""
        return self.config.num_classes

    @property
    def num_levels(self) -> int:
        """Return encoded feature level count."""
        return self.config.num_levels

    def forward(  # noqa: PLR0915
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        *,
        plan: DecoderPlan | None = None,
    ) -> DistributionOutput:
        """Run the requested prefix and return FDR predictions and distributions."""
        queries.validate()
        features.validate()
        if queries.content.shape[0] != features.memory.shape[0]:
            msg = "Queries and memory must share batch size"
            raise ValueError(msg)
        if (
            queries.content.dtype != features.memory.dtype
            or queries.content.device != features.memory.device
        ):
            msg = "Queries and memory must share dtype and device"
            raise ValueError(msg)
        if (
            queries.content.shape[2] != self.config.hidden_dim
            or features.memory.shape[2] != self.config.hidden_dim
        ):
            msg = "Queries and memory must match decoder hidden dimension"
            raise ValueError(msg)
        if features.spatial_shapes.shape[0] != self.config.num_levels:
            msg = "Encoded feature levels must match decoder configuration"
            raise ValueError(msg)
        selected = (
            self.default_eval_plan
            if plan is None and not self.training
            else (plan or DecoderPlan())
        )
        depth = self.config.num_layers if selected.num_layers is None else selected.num_layers
        if not 1 <= depth <= self.config.num_layers:
            msg = "Decoder plan num_layers must select a configured prefix"
            raise ValueError(msg)

        output = queries.content
        sampling_reference = queries.ref_logits.sigmoid()
        previous_output = torch.zeros_like(output)
        previous_corners = torch.zeros(
            *output.shape[:2], 4 * (self.reg_max + 1), dtype=output.dtype, device=output.device
        )
        initial_references = sampling_reference
        logits, boxes, corners = [], [], []
        pre = None
        for index, layer in enumerate(self.layers[:depth]):
            query_position = self.query_pos_head(sampling_reference).clamp(-10, 10)
            output = layer(
                output,
                sampling_reference.unsqueeze(2),
                features.memory,
                features.spatial_shapes,
                query_position,
                queries.attention_mask,
            )
            if index == 0:
                pre_boxes = (
                    self.pre_bbox_head(output) + inverse_sigmoid(sampling_reference)
                ).sigmoid()
                pre = DetectionOutput(self.score_heads[0](output), pre_boxes)
                initial_references = pre_boxes.detach()
            current_corners = (
                self.distribution_heads[index](output + previous_output.detach()) + previous_corners
            )
            corner_distribution = current_corners.reshape(
                *current_corners.shape[:2], 4, self.reg_max + 1
            )
            bins = weighting_function(self.reg_max, self.up, self.reg_scale, corner_distribution)
            distances = functional.softmax(corner_distribution, dim=-1).matmul(bins)
            refined_boxes = distance_to_boxes(initial_references, distances, self.reg_scale)
            current_scores = self.score_heads[index](output)
            current_logits = (
                self.lqe[index](current_scores, corner_distribution)
                if self.use_lqe
                else current_scores
            )
            if selected.collect_intermediate or index == depth - 1:
                logits.append(current_logits)
                boxes.append(refined_boxes)
                corners.append(
                    current_corners.reshape(*current_corners.shape[:2], 4, self.reg_max + 1)
                )
            previous_corners = current_corners
            previous_output = output
            sampling_reference = refined_boxes.detach()
        if pre is None:
            msg = "FDR decoder executed no layers"
            raise RuntimeError(msg)
        result = DistributionOutput(
            torch.stack(logits),
            torch.stack(boxes),
            torch.stack(corners),
            initial_references,
            self.reg_scale,
            self.up,
            pre,
        )
        result.validate()
        return result

    def checkpoint_config(self) -> dict[str, object]:
        """Record baseline decoder and all FDR construction settings."""
        return {
            "config": asdict(self.config),
            "default_eval_plan": asdict(self.default_eval_plan),
            "reg_max": self.reg_max,
            "reg_scale": self.reg_scale,
            "up": self.up,
            "use_lqe": self.use_lqe,
        }
