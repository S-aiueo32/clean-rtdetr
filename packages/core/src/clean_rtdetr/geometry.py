# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR geometry, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: explicit arguments, validation, and standalone anchor generation.
Provenance and the upstream license are recorded in third_party/RT-DETR/.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

MAX_PROBABILITY_EPS = 0.5


def inverse_sigmoid(values: Float[Tensor, "*shape"], eps: float = 1e-5) -> Float[Tensor, "*shape"]:
    """Clamp values to the unit interval, then apply upstream endpoint stabilization.

    The numerator and denominator use separate ``eps`` clamps before the log-ratio.
    """
    if not 0 < eps < MAX_PROBABILITY_EPS:
        msg = "eps must be between 0 and 0.5"
        raise ValueError(msg)
    values = values.clamp(0, 1)
    return torch.log(values.clamp(min=eps) / (1 - values).clamp(min=eps))


def generate_anchors(
    spatial_shapes: tuple[tuple[int, int], ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    grid_size: float = 0.05,
    eps: float = 0.01,
) -> tuple[Float[Tensor, "1 sequence 4"], Bool[Tensor, "1 sequence 1"]]:
    """Return logit cxcywh [1,S,4] and True=valid bool [1,S,1].

    Reference anchors are generated on CPU at ``dtype`` and then moved to
    ``device``, matching upstream arithmetic. Invalid anchor logits intentionally
    contain +inf.
    """
    if not spatial_shapes or not 0 < eps < MAX_PROBABILITY_EPS or not 0 < grid_size < 1:
        msg = "Expected nonempty shapes, 0 < eps < .5 and 0 < grid_size < 1"
        raise ValueError(msg)
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        if height <= 0 or width <= 0:
            msg = "Anchor dimensions must be positive"
            raise ValueError(msg)
        y, x = torch.meshgrid(
            torch.arange(height, dtype=dtype),
            torch.arange(width, dtype=dtype),
            indexing="ij",
        )
        centers = (torch.stack((x, y), -1).unsqueeze(0) + 0.5) / torch.tensor(
            [width, height], dtype=dtype
        )
        sizes = torch.ones_like(centers) * grid_size * (2.0**level)
        anchors.append(torch.cat((centers, sizes), -1).reshape(1, height * width, 4))
    boxes = torch.cat(anchors, 1).to(device)
    valid = ((boxes > eps) & (boxes < 1 - eps)).all(-1, keepdim=True)
    logits = torch.log(boxes / (1 - boxes))
    return torch.where(valid, logits, torch.inf), valid
