# Copyright 2026 Clean RT-DETR contributors
"""Bounding-box geometry adapted from the pinned RT-DETR implementation.

The upstream implementation is vendored under ``third_party/RT-DETR`` and is
Apache-2.0 licensed; this module adds typed contracts and standalone helpers.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def cxcywh_to_xyxy(boxes: Float[Tensor, " boxes 4"]) -> Float[Tensor, " boxes 4"]:
    """Convert center-coordinate boxes to corner-coordinate boxes."""
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def box_iou(
    boxes1: Float[Tensor, " boxes1 4"], boxes2: Float[Tensor, " boxes2 4"]
) -> tuple[Float[Tensor, " boxes1 boxes2"], Float[Tensor, " boxes1 boxes2"]]:
    """Compute pairwise IoU for corner-coordinate boxes."""
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(-1)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(
    boxes1: Float[Tensor, " boxes1 4"], boxes2: Float[Tensor, " boxes2 4"]
) -> Float[Tensor, " boxes1 boxes2"]:
    """Compute pairwise generalized IoU for corner-coordinate boxes.

    This follows the arithmetic in the pinned upstream ``box_ops.py``.
    """
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area = (rb - lt).clamp(min=0).prod(-1)
    return iou - (area - union) / area
