# Copyright 2026 Clean RT-DETR contributors
"""Contracts and validation for training targets."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int64, Shaped
from torch import Tensor

from clean_rtdetr.core.contracts import BOX_COORDINATES, QUERY_RANK, DetectionOutput

_ALLOWED_FLOAT_DTYPES = (torch.float32, torch.float64)
_TARGET_BOX_RANK = 2


@dataclass(frozen=True)
class DetectionTarget:
    """One image's labels and normalized target boxes."""

    labels: Int64[Tensor, " targets"]
    boxes: Float[Tensor, " targets 4"]


@dataclass(frozen=True)
class MatchIndices:
    """Prediction and target indices selected by bipartite matching."""

    prediction_indices: Int64[Tensor, " matches"]
    target_indices: Int64[Tensor, " matches"]


def _validate_predictions(output: DetectionOutput) -> tuple[int, int, int]:
    logits, boxes = output.logits, output.boxes
    _validate_prediction_shapes(logits, boxes)
    _validate_prediction_values(logits, boxes)
    return int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2])


def _validate_prediction_shapes(
    logits: Shaped[Tensor, "..."], boxes: Shaped[Tensor, "..."]
) -> None:
    if logits.ndim != QUERY_RANK or boxes.ndim != QUERY_RANK:
        msg = "Detection logits and boxes must have rank 3"
        raise ValueError(msg)
    if min(logits.shape) <= 0:
        msg = "Detection logits must have nonempty shape [B, Q, C]"
        raise ValueError(msg)
    if boxes.shape != (logits.shape[0], logits.shape[1], BOX_COORDINATES):
        msg = "Detection boxes must have shape [B, Q, 4]"
        raise ValueError(msg)
    if logits.dtype not in _ALLOWED_FLOAT_DTYPES or boxes.dtype not in _ALLOWED_FLOAT_DTYPES:
        msg = "Detection tensors must use float32 or float64"
        raise ValueError(msg)
    if logits.dtype != boxes.dtype or logits.device != boxes.device:
        msg = "Detection logits and boxes must share floating dtype and device"
        raise ValueError(msg)


def _validate_prediction_values(
    logits: Shaped[Tensor, "..."], boxes: Shaped[Tensor, "..."]
) -> None:
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        msg = "Detection logits and boxes must be finite"
        raise ValueError(msg)
    if bool((boxes < 0).any()) or bool((boxes > 1).any()):
        msg = "Detection boxes must be normalized to [0, 1]"
        raise ValueError(msg)


def _validate_target(
    target: DetectionTarget, *, classes: int, dtype: torch.dtype, device: torch.device
) -> None:
    labels, boxes = target.labels, target.boxes
    _validate_target_structure(labels, boxes)
    _validate_target_compatibility(labels, boxes, dtype=dtype, device=device)
    _validate_target_values(labels, boxes, classes=classes)


def _validate_target_structure(labels: Shaped[Tensor, "..."], boxes: Shaped[Tensor, "..."]) -> None:
    if labels.ndim != 1 or labels.dtype != torch.int64:
        msg = "Target labels must be one-dimensional int64 tensors"
        raise ValueError(msg)
    if boxes.ndim != _TARGET_BOX_RANK or boxes.shape[1] != BOX_COORDINATES:
        msg = "Target boxes must have shape [N, 4]"
        raise ValueError(msg)
    if boxes.shape[0] != labels.shape[0]:
        msg = "Target labels and boxes must have matching lengths"
        raise ValueError(msg)


def _validate_target_compatibility(
    labels: Shaped[Tensor, "..."],
    boxes: Shaped[Tensor, "..."],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if boxes.dtype != dtype or boxes.dtype not in _ALLOWED_FLOAT_DTYPES:
        msg = "Target boxes must share prediction dtype and use float32 or float64"
        raise ValueError(msg)
    if boxes.device != device or labels.device != device:
        msg = "Targets must be on the prediction device"
        raise ValueError(msg)


def _validate_target_values(
    labels: Shaped[Tensor, "..."], boxes: Shaped[Tensor, "..."], *, classes: int
) -> None:
    if labels.numel() and (bool((labels < 0).any()) or bool((labels >= classes).any())):
        msg = "Target labels must be in the prediction class range"
        raise ValueError(msg)
    if not torch.isfinite(boxes).all():
        msg = "Target boxes must be finite"
        raise ValueError(msg)
    if bool((boxes < 0).any()) or bool((boxes > 1).any()):
        msg = "Target boxes must be normalized to [0, 1]"
        raise ValueError(msg)
    if boxes.numel() and bool((boxes[:, 2:] <= 0).any()):
        msg = "Target box widths and heights must be positive"
        raise ValueError(msg)


def validate_detection(output: DetectionOutput, targets: Sequence[DetectionTarget]) -> None:
    """Validate predictions and their per-image normalized detection targets."""
    batch, _, classes = _validate_predictions(output)
    if len(targets) != batch:
        msg = "Expected one target per batch image"
        raise ValueError(msg)
    for target in targets:
        _validate_target(
            target, classes=classes, dtype=output.logits.dtype, device=output.logits.device
        )
