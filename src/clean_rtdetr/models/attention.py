# Copyright 2026 Clean RT-DETR contributors
"""Multi-scale deformable attention adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: tensor metadata input, explicit validation, and a standalone exchange point.
See third_party/RT-DETR/README.md.
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.core.contracts import BOX_COORDINATES, SPATIAL_DIMENSIONS


def deformable_attention_core(
    value: Float[Tensor, "batch sequence heads head_dim"],
    spatial_shapes: Tensor,
    sampling_locations: Float[Tensor, "batch queries heads levels points coordinates"],
    attention_weights: Float[Tensor, "batch queries heads levels points"],
) -> Float[Tensor, "batch queries channels"]:
    """Sample flattened feature levels with upstream bilinear-grid semantics."""
    batch, _, num_heads, head_dim = value.shape
    _, queries, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = tuple((int(height), int(width)) for height, width in spatial_shapes.tolist())
    values_by_level = value.split([height * width for height, width in shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(shapes):
        level_value = (
            values_by_level[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * num_heads, head_dim, height, width)
        )
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            functional.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * num_heads, 1, queries, num_levels * num_points
    )
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.reshape(batch, num_heads * head_dim, queries).transpose(1, 2)


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable cross-attention with the upstream parameter layout."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ) -> None:
        """Construct projections and initialize sampling offsets radially."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads
        total_points = num_heads * num_levels * num_points
        self.sampling_offsets = nn.Linear(hidden_dim, total_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, total_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Match the radial offset and projection initialization used upstream."""
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        directions = directions / directions.abs().max(-1, keepdim=True).values
        directions = directions.reshape(self.num_heads, 1, 1, 2).tile(
            1, self.num_levels, self.num_points, 1
        )
        scales = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_((directions * scales).flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Float[Tensor, "batch queries channels"],
        reference_points: Float[Tensor, "batch queries levels coordinates"],
        value: Float[Tensor, "batch sequence channels"],
        spatial_shapes: Tensor,
    ) -> Float[Tensor, "batch queries channels"]:
        """Attend to memory around normalized two- or four-coordinate references."""
        batch, queries, _ = query.shape
        sequence = value.shape[1]
        if spatial_shapes.shape != (self.num_levels, SPATIAL_DIMENSIONS):
            msg = "Spatial shapes must match the attention level count"
            raise ValueError(msg)
        if int(spatial_shapes.prod(dim=1).sum()) != sequence:
            msg = "Memory length must equal the summed attention level areas"
            raise ValueError(msg)
        value = self.value_proj(value).reshape(batch, sequence, self.num_heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(
            batch,
            queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            SPATIAL_DIMENSIONS,
        )
        weights = self.attention_weights(query).reshape(
            batch, queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = functional.softmax(weights, dim=-1).reshape(
            batch, queries, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == SPATIAL_DIMENSIONS:
            normalizer = (
                spatial_shapes.flip(1)
                .to(dtype=query.dtype)
                .reshape(1, 1, 1, self.num_levels, 1, SPATIAL_DIMENSIONS)
            )
            locations = (
                reference_points.reshape(batch, queries, 1, self.num_levels, 1, SPATIAL_DIMENSIONS)
                + offsets / normalizer
            )
        elif reference_points.shape[-1] == BOX_COORDINATES:
            locations = (
                reference_points[:, :, None, :, None, :SPATIAL_DIMENSIONS]
                + offsets
                / self.num_points
                * reference_points[:, :, None, :, None, SPATIAL_DIMENSIONS:]
                * 0.5
            )
        else:
            msg = "Reference points must end in two or four coordinates"
            raise ValueError(msg)
        return self.output_proj(
            deformable_attention_core(value, spatial_shapes, locations, weights)
        )
