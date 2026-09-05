# Copyright 2026 Clean RT-DETR contributors
"""Convert decoder predictions into ranked, image-scale detections.

The ranking and box conversion follow RT-DETR's inference operation order;
thresholding, clipping, and NMS remain intentionally outside this boundary.

Adapted from lyuwenyu/RT-DETR (Apache-2.0). Changed: explicit typed result
and eager boundary validation. See ``third_party/RT-DETR`` for provenance.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int64, Shaped
from torch import Tensor

from clean_rtdetr.core.contracts import DetectionOutput

LOGITS_RANK = 3
BOX_RANK = 3
BOX_COORDINATES = 4
SIZE_RANK = 2
SIZE_DIMENSIONS = 2


@dataclass(frozen=True)
class Detections:
    """Top-scoring detections, represented in image-space ``xyxy`` coordinates."""

    labels: Int64[Tensor, "batch top_k"]
    scores: Float[Tensor, "batch top_k"]
    boxes_xyxy: Float[Tensor, "batch top_k 4"]


def _validate_inputs(
    predictions: DetectionOutput, original_sizes_wh: Shaped[Tensor, "batch dimensions"]
) -> tuple[int, int, int]:
    """Validate postprocessing inputs and return batch/query/class dimensions."""
    logits = predictions.logits
    boxes = predictions.boxes
    if logits.ndim != LOGITS_RANK or min(logits.shape) <= 0:
        msg = "Logits must be nonempty with shape [B, Q, K]"
        raise ValueError(msg)
    if boxes.ndim != BOX_RANK or boxes.shape[-1] != BOX_COORDINATES or min(boxes.shape) <= 0:
        msg = "Boxes must be nonempty with shape [B, Q, 4]"
        raise ValueError(msg)
    if not logits.is_floating_point() or not boxes.is_floating_point():
        msg = "Logits and boxes must be floating-point tensors"
        raise ValueError(msg)
    if logits.shape[:2] != boxes.shape[:2] or logits.dtype != boxes.dtype:
        msg = "Logits and boxes must share [B, Q] dimensions and dtype"
        raise ValueError(msg)
    if logits.device != boxes.device:
        msg = "Logits and boxes must be on the same device"
        raise ValueError(msg)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(boxes).all()):
        msg = "Logits and boxes must contain only finite values"
        raise ValueError(msg)
    batch, queries, classes = logits.shape
    if original_sizes_wh.ndim != SIZE_RANK or original_sizes_wh.shape != (batch, SIZE_DIMENSIONS):
        msg = "Original sizes must have shape [B, 2]"
        raise ValueError(msg)
    if original_sizes_wh.dtype != torch.int64:
        msg = "Original sizes must be int64"
        raise ValueError(msg)
    if not bool((original_sizes_wh > 0).all()):
        msg = "Original sizes must be positive"
        raise ValueError(msg)
    return batch, queries, classes


def postprocess(
    predictions: DetectionOutput,
    original_sizes_wh: Shaped[Tensor, "batch dimensions"],
    *,
    top_k: int = 300,
) -> Detections:
    """Rank multi-label decoder predictions and scale boxes to original images.

    Predictions are flattened over query and class, so one query may contribute
    multiple detections. No score thresholding, clipping, or NMS is performed.
    """
    logits = predictions.logits
    boxes = predictions.boxes
    batch, queries, classes = _validate_inputs(predictions, original_sizes_wh)
    if type(top_k) is not int or not 1 <= top_k <= queries * classes:
        msg = "top_k must be an integer in [1, Q*K]"
        raise ValueError(msg)

    scores, flat_indices = torch.topk(
        logits.sigmoid().reshape(batch, queries * classes), top_k, dim=1, sorted=True
    )
    query_indices = torch.div(flat_indices, classes, rounding_mode="floor")
    labels = flat_indices.remainder(classes).to(torch.int64)

    cx, cy, width, height = boxes.unbind(dim=-1)
    boxes_xyxy = torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=-1
    )
    sizes = original_sizes_wh.to(device=boxes.device, dtype=boxes.dtype)
    scale = sizes[:, (0, 1, 0, 1)].unsqueeze(1)
    boxes_xyxy = boxes_xyxy * scale
    boxes_xyxy = boxes_xyxy.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 4))
    return Detections(labels=labels, scores=scores, boxes_xyxy=boxes_xyxy)
