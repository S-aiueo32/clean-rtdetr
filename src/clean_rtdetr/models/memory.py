"""Decoder input projection, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: standalone module, FeatureSpec-driven construction and typed output.
"""

from collections import OrderedDict

import torch
from torch import nn

from clean_rtdetr.core.contracts import EncodedFeatures, FeatureSpec, MultiScaleFeatures


class MemoryProjector(nn.Module):
    """Project supplied levels to hidden_dim and flatten without implicit extra levels."""

    def __init__(self, specs: tuple[FeatureSpec, ...], hidden_dim: int = 256):
        super().__init__()
        if not specs or hidden_dim <= 0:
            raise ValueError("Expected feature specs and a positive hidden dimension")
        if any(left.stride >= right.stride for left, right in zip(specs, specs[1:], strict=False)):
            raise ValueError("Feature specs must have strictly increasing strides")
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
        features.validate()
        if features.specs != self.specs:
            raise ValueError("Input FeatureSpec differs from the projector's construction spec")
        projected = [
            projection(level)
            for projection, level in zip(self.input_proj, features.levels, strict=True)
        ]
        memory = torch.cat([level.flatten(2).transpose(1, 2) for level in projected], dim=1)
        shapes = torch.tensor(
            [level.shape[-2:] for level in projected], dtype=torch.int64, device=memory.device
        )
        sizes = shapes.prod(dim=1)
        return EncodedFeatures(memory, shapes, sizes.cumsum(0) - sizes)
