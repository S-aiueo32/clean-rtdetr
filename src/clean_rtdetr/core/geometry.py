"""RT-DETR geometry, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: explicit arguments, validation, and standalone anchor generation.
Provenance and the upstream license are recorded in third_party/RT-DETR/.
"""

import torch
from torch import Tensor


def inverse_sigmoid(values: Tensor, eps: float = 1e-5) -> Tensor:
    """Preserve upstream's separate numerator/denominator clamp at the endpoints."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    values = values.clamp(0, 1)
    return torch.log(values.clamp(min=eps) / (1 - values).clamp(min=eps))


def generate_anchors(
    spatial_shapes: tuple[tuple[int, int], ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    grid_size: float = 0.05,
    eps: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Return logit cxcywh [1,S,4] and True=valid bool [1,S,1].

    Reference anchors are generated on CPU then moved, matching upstream's
    arithmetic. Shape specialization and lower precision are later deployment work.
    Invalid anchor logits intentionally contain +inf.
    """
    if not spatial_shapes or not 0 < eps < 0.5 or not 0 < grid_size < 1:
        raise ValueError("Expected nonempty shapes, 0 < eps < .5 and 0 < grid_size < 1")
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        if height <= 0 or width <= 0:
            raise ValueError("Anchor dimensions must be positive")
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
