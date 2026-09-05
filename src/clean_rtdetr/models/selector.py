"""RT-DETR query selection, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: extracted from RTDETRTransformer; typed boundaries; head injection;
query reference detach deferred to assembly. See third_party/RT-DETR/README.md.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from clean_rtdetr.core.contracts import EncodedFeatures, QuerySet, SelectionOutput
from clean_rtdetr.core.geometry import generate_anchors


@dataclass(frozen=True)
class QuerySelectorConfig:
    hidden_dim: int = 256
    num_classes: int = 80
    num_queries: int = 300
    learned_content: bool = False
    anchor_eps: float = 0.01

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.num_classes, self.num_queries) <= 0:
            raise ValueError("Hidden dimension, class count, and query count must be positive")
        if not 0 < self.anchor_eps < 0.5:
            raise ValueError("Anchor epsilon must be between 0 and 0.5")


class BoxHead(nn.Module):
    """Three-layer ReLU MLP, retaining upstream key layout for stage weight loading."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 4),
            ]
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, values: Tensor) -> Tensor:
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
    ):
        super().__init__()
        self.config = config
        self.enc_output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim)
        )
        nn.init.xavier_uniform_(self.enc_output[0].weight)
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
        features.validate()
        memory = features.memory
        batch, sequence, channels = memory.shape
        cfg = self.config
        if channels != cfg.hidden_dim:
            raise ValueError("Memory channels do not match the selector hidden dimension")
        if sequence < cfg.num_queries:
            raise ValueError("Memory sequence length must be at least num_queries")
        shapes = tuple(tuple(shape) for shape in features.spatial_shapes.tolist())
        anchors, valid = generate_anchors(shapes, device=memory.device, eps=cfg.anchor_eps)
        # Upstream masks memory, not scores: LayerNorm/linear biases still contribute.
        encoded = self.enc_output(memory * valid.to(memory.dtype))
        logits = self.enc_score_head(encoded)
        deltas = self.enc_bbox_head(encoded)
        if logits.shape != (batch, sequence, cfg.num_classes):
            raise ValueError("Score head must return [B, S, num_classes]")
        if deltas.shape != (batch, sequence, 4):
            raise ValueError("Box head must return [B, S, 4] coordinate deltas")
        coordinates = deltas + anchors
        indices = torch.topk(logits.max(-1).values, cfg.num_queries, dim=1).indices

        def gather(values: Tensor) -> Tensor:
            return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

        refs = gather(coordinates)
        content = (
            self.tgt_embed.weight.unsqueeze(0).expand(batch, -1, -1)
            if self.tgt_embed is not None
            else gather(encoded).detach()
        )
        return SelectionOutput(QuerySet(content, refs, indices), gather(logits), refs.sigmoid())
