# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR query selection, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: extracted from RTDETRTransformer; typed boundaries; head injection;
query reference detach deferred to assembly. See third_party/RT-DETR/README.md.
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import EncodedFeatures, QuerySet, SelectionOutput
from clean_rtdetr.core.geometry import MAX_PROBABILITY_EPS, generate_anchors


@dataclass(frozen=True)
class QuerySelectorConfig:
    """Define RT-DETR query dimensions, initialization, and anchor validity."""

    hidden_dim: int = 256
    num_classes: int = 80
    num_queries: int = 300
    learned_content: bool = False
    anchor_eps: float = 0.01

    def __post_init__(self) -> None:
        """Reject invalid dimensions and probability margins at construction."""
        if min(self.hidden_dim, self.num_classes, self.num_queries) <= 0:
            msg = "Hidden dimension, class count, and query count must be positive"
            raise ValueError(msg)
        if not 0 < self.anchor_eps < MAX_PROBABILITY_EPS:
            msg = "Anchor epsilon must be between 0 and 0.5"
            raise ValueError(msg)


class BoxHead(nn.Module):
    """Three-layer ReLU MLP, retaining upstream key layout for stage weight loading."""

    def __init__(self, hidden_dim: int) -> None:
        """Build the upstream MLP and zero-initialize its final regression layer."""
        super().__init__()
        layers = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 4),
        ]
        self.layers = nn.ModuleList(layers)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, values: Float[Tensor, "*batch channels"]) -> Float[Tensor, "*batch 4"]:
        """Predict four coordinate deltas for each memory position."""
        for layer in self.layers[:-1]:
            values = torch.relu(layer(values))
        return self.layers[-1](values)


class QuerySelector(nn.Module):
    """Select top-k encoder positions; keep upstream masking and gradient semantics.

    Custom score/box heads are registered as child modules and keep their own
    initialization. They must return [B,S,K] and [B,S,4] respectively. A variant
    changing the state representation should replace the selector boundary too.
    """

    def __init__(
        self,
        config: QuerySelectorConfig,
        *,
        score_head: nn.Module | None = None,
        box_head: nn.Module | None = None,
    ) -> None:
        """Construct default heads or register the supplied research modules."""
        super().__init__()
        self.config = config
        projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.enc_output = nn.Sequential(projection, nn.LayerNorm(config.hidden_dim))
        nn.init.xavier_uniform_(projection.weight)
        if score_head is None:
            score_head = nn.Linear(config.hidden_dim, config.num_classes)
            nn.init.constant_(score_head.bias, -math.log(99.0))
        self.enc_score_head = score_head
        self.enc_bbox_head = box_head if box_head is not None else BoxHead(config.hidden_dim)
        self.tgt_embed = (
            nn.Embedding(config.num_queries, config.hidden_dim) if config.learned_content else None
        )
        if self.tgt_embed is not None:
            nn.init.xavier_uniform_(self.tgt_embed.weight)

    def forward(self, features: EncodedFeatures) -> SelectionOutput:
        """Select positions and preserve the separate query/auxiliary gradient routes."""
        features.validate()
        memory = features.memory
        _, sequence, channels = memory.shape
        cfg = self.config
        if channels != cfg.hidden_dim:
            msg = "Memory channels do not match the selector hidden dimension"
            raise ValueError(msg)
        if sequence < cfg.num_queries:
            msg = "Memory sequence length must be at least num_queries"
            raise ValueError(msg)
        shapes = tuple(tuple(shape) for shape in features.spatial_shapes.tolist())
        anchors, valid = generate_anchors(shapes, device=memory.device, eps=cfg.anchor_eps)
        return self._select(memory, anchors, valid)

    def _select(
        self,
        memory: Float[Tensor, "batch sequence channels"],
        anchors: Float[Tensor, "1 sequence 4"],
        valid: Bool[Tensor, "1 sequence 1"],
    ) -> SelectionOutput:
        """Share selection arithmetic with validated, shape-specialized callers."""
        batch, sequence, _ = memory.shape
        cfg = self.config
        # Upstream masks memory, not scores: LayerNorm/linear biases still contribute.
        encoded = self.enc_output(memory * valid.to(memory.dtype))
        logits = self.enc_score_head(encoded)
        deltas = self.enc_bbox_head(encoded)
        if logits.shape != (batch, sequence, cfg.num_classes):
            msg = "Score head must return [B, S, num_classes]"
            raise ValueError(msg)
        if deltas.shape != (batch, sequence, 4):
            msg = "Box head must return [B, S, 4] coordinate deltas"
            raise ValueError(msg)
        coordinates = deltas + anchors
        indices = torch.topk(logits.max(-1).values, cfg.num_queries, dim=1).indices

        def gather(
            values: Float[Tensor, "batch sequence channels"],
        ) -> Float[Tensor, "batch queries channels"]:
            return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

        refs = gather(coordinates)
        content = (
            self.tgt_embed.weight.unsqueeze(0).expand(batch, -1, -1)
            if self.tgt_embed is not None
            else gather(encoded).detach()
        )
        return SelectionOutput(QuerySet(content, refs, indices), gather(logits), refs.sigmoid())
