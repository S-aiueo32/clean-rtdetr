# Copyright 2026 Clean RT-DETR contributors
"""Decoder input projection, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: standalone module, FeatureSpec-driven construction and typed output.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict
from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from jaxtyping import Float, Int64

from clean_rtdetr.contracts import EncodedFeatures, FeatureSpec, MultiScaleFeatures
from clean_rtdetr.models.interfaces import Projector


class MemoryProjector(Projector):
    """Project supplied levels to hidden_dim and flatten without implicit extra levels."""

    def __init__(self, specs: Sequence[FeatureSpec], hidden_dim: int = 256) -> None:
        """Build a Conv/BN projection for each explicitly supplied feature level."""
        super().__init__()
        if not specs or hidden_dim <= 0:
            msg = "Expected feature specs and a positive hidden dimension"
            raise ValueError(msg)
        if any(left.stride >= right.stride for left, right in pairwise(specs)):
            msg = "Feature specs must have strictly increasing strides"
            raise ValueError(msg)
        self.specs = tuple(specs)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.ModuleList(
            nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(spec.channels, hidden_dim, 1, bias=False),
                    norm=nn.BatchNorm2d(hidden_dim),
                )
            )
            for spec in specs
        )

    def forward(self, features: MultiScaleFeatures) -> EncodedFeatures:
        """Project features and concatenate their row-major spatial positions."""
        features.validate()
        if features.specs != self.specs:
            msg = "Input FeatureSpec differs from the projector's construction spec"
            raise ValueError(msg)
        projected = [
            projection(level)
            for projection, level in zip(self.input_proj, features.levels, strict=True)
        ]
        memory: Float[Tensor, "batch sequence channels"] = torch.cat(
            [level.flatten(2).transpose(1, 2) for level in projected], dim=1
        )
        shapes: Int64[Tensor, "levels 2"] = torch.tensor(
            [level.shape[-2:] for level in projected], dtype=torch.int64, device=memory.device
        )
        sizes = shapes.prod(dim=1)
        return EncodedFeatures(memory, shapes, sizes.cumsum(0) - sizes)

    def checkpoint_config(self) -> dict[str, object]:
        """Describe projector dimensions and input feature specifications."""
        return {"specs": tuple(asdict(spec) for spec in self.specs), "hidden_dim": self.hidden_dim}
