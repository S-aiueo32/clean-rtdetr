# Copyright 2026 Clean RT-DETR contributors
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""D-FINE's nonuniform bins and box transforms, adapted under Apache-2.0.

See ../NOTICE.md for the pinned source. Constants are caller-owned floats;
generated bins inherit the reference tensor's dtype and device.
"""

import math

import torch
from jaxtyping import Float, Int64
from torch import Tensor


def weighting_function(
    reg_max: int, up: float, reg_scale: float, reference: Float[Tensor, "..."]
) -> Float[Tensor, " bins"]:
    """Return the symmetric, nonuniform reg_max+1 projection coordinates."""
    minimum_bins = 4
    if type(reg_max) is not int or reg_max < minimum_bins or reg_max % 2:
        msg = "reg_max must be an even integer of at least four"
        raise ValueError(msg)
    if not all(math.isfinite(value) and value > 0 for value in (up, reg_scale)):
        msg = "up and reg_scale must be finite and positive"
        raise ValueError(msg)
    bound = up * reg_scale
    step = (bound + 1) ** (2 / (reg_max - 2))
    powers = torch.arange(1, reg_max // 2, dtype=reference.dtype, device=reference.device)
    positive = step**powers - 1
    return torch.cat(
        (
            reference.new_tensor([-2 * bound]),
            -positive.flip(0),
            reference.new_zeros(1),
            positive,
            reference.new_tensor([2 * bound]),
        )
    )


def distance_to_boxes(
    references: Float[Tensor, "... 4"],
    distances: Float[Tensor, "... 4"],
    reg_scale: float,
) -> Float[Tensor, "... 4"]:
    """Decode signed left/top/right/bottom offsets into un-clipped cxcywh."""
    scale = references[..., 2:] / reg_scale
    low = references[..., :2] - (0.5 * reg_scale + distances[..., :2]) * scale
    high = references[..., :2] + (0.5 * reg_scale + distances[..., 2:]) * scale
    return torch.cat(((low + high) * 0.5, high - low), dim=-1)


def box_to_distances(
    references: Float[Tensor, "... 4"],
    boxes_xyxy: Float[Tensor, "... 4"],
    reg_scale: float,
) -> Float[Tensor, "... 4"]:
    """Express target corners in the fixed initial reference coordinate frame."""
    scale = references[..., 2:] / reg_scale + 1e-16
    low = (references[..., :2] - boxes_xyxy[..., :2]) / scale - 0.5 * reg_scale
    high = (boxes_xyxy[..., 2:] - references[..., :2]) / scale - 0.5 * reg_scale
    return torch.cat((low, high), dim=-1)


def interpolate_bins(
    distances: Float[Tensor, "..."], bins: Float[Tensor, " bins"]
) -> tuple[Int64[Tensor, "..."], Float[Tensor, "..."]]:
    """Return left neighbor and right weight, saturating at the end bins.

    At or beyond the highest bin, use the final interval with right weight one.
    At or below the lowest bin, use the first interval with right weight zero.
    """
    values = distances.detach().contiguous()
    left = (torch.searchsorted(bins, values, right=True) - 1).clamp(0, bins.numel() - 2)
    right_weight = ((values - bins[left]) / (bins[left + 1] - bins[left])).clamp(0, 1)
    return left, right_weight
