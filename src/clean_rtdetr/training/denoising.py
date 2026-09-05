# Copyright 2026 Clean RT-DETR contributors
"""Construct contrastive denoising queries for RT-DETR training.

Adapted from lyuwenyu/RT-DETR (Apache-2.0); the pinned source and license are
recorded under ``third_party/RT-DETR``. This version adds typed contracts,
validation, and float64 support while retaining upstream FP32 RNG ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float  # noqa: TC002 - runtime jaxtyping instrumentation resolves annotations
from torch import Tensor, nn

from clean_rtdetr.core.contracts import DenoisingMetadata, DenoisingQueries
from clean_rtdetr.core.geometry import inverse_sigmoid
from clean_rtdetr.training.contracts import DetectionTarget, _validate_target

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class DenoisingConfig:
    """Noise levels and requested target repetitions for DN training."""

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject settings outside the upstream operation's useful domain."""
        if (
            isinstance(self.num_denoising, bool)
            or not isinstance(self.num_denoising, int)
            or self.num_denoising < 0
        ):
            msg = "num_denoising must be a nonnegative integer"
            raise ValueError(msg)
        if not math.isfinite(self.label_noise_ratio) or not 0 <= self.label_noise_ratio <= 1:
            msg = "label_noise_ratio must be between 0 and 1"
            raise ValueError(msg)
        if not math.isfinite(self.box_noise_scale) or self.box_noise_scale < 0:
            msg = "box_noise_scale must be nonnegative"
            raise ValueError(msg)


@dataclass(frozen=True)
class DenoisingBatch:
    """Prepared decoder queries and the indices needed by DN losses."""

    queries: DenoisingQueries
    metadata: DenoisingMetadata


def _box_cxcywh_to_xyxy(
    boxes: Float[Tensor, "*batch 4"],
) -> Float[Tensor, "*batch 4"]:
    """Convert center-size boxes to corner coordinates."""
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _box_xyxy_to_cxcywh(
    boxes: Float[Tensor, "*batch 4"],
) -> Float[Tensor, "*batch 4"]:
    """Convert corner boxes to center-size coordinates."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


class DenoisingQueryGenerator(nn.Module):
    """Build padded positive/negative target groups with upstream RNG ordering."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        config: DenoisingConfig | None = None,
    ) -> None:
        """Create the padding-aware class embedding and store validated dimensions."""
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            msg = "num_classes must be a positive integer"
            raise ValueError(msg)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            msg = "hidden_dim must be a positive integer"
            raise ValueError(msg)
        if config is not None and not isinstance(config, DenoisingConfig):
            msg = "config must be a DenoisingConfig"
            raise TypeError(msg)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.config = config if config is not None else DenoisingConfig()
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    def forward(  # noqa: C901, PLR0915
        self,
        targets: Sequence[DetectionTarget],
        *,
        num_queries: int,
    ) -> DenoisingBatch | None:
        """Return DN queries, or ``None`` when disabled or the whole batch is empty."""
        if isinstance(num_queries, bool) or not isinstance(num_queries, int) or num_queries <= 0:
            msg = "num_queries must be a positive integer"
            raise ValueError(msg)
        if not targets:
            msg = "Denoising targets must contain at least one batch image"
            raise ValueError(msg)

        weight = self.class_embed.weight
        for target in targets:
            if not isinstance(target, DetectionTarget):
                msg = "targets must contain DetectionTarget values"
                raise TypeError(msg)
            _validate_target(
                target,
                classes=self.num_classes,
                dtype=weight.dtype,
                device=weight.device,
            )

        cfg = self.config
        if cfg.num_denoising == 0:
            return None
        target_counts = [int(target.labels.shape[0]) for target in targets]
        max_targets = max(target_counts)
        if max_targets == 0:
            return None

        device, dtype = weight.device, weight.dtype
        num_groups = max(1, cfg.num_denoising // max_targets)
        batch_size = len(targets)
        query_classes = torch.full(
            (batch_size, max_targets), self.num_classes, dtype=torch.int32, device=device
        )
        query_boxes = torch.zeros((batch_size, max_targets, 4), dtype=dtype, device=device)
        padding_mask = torch.zeros((batch_size, max_targets), dtype=torch.bool, device=device)
        for image_index, target in enumerate(targets):
            count = target_counts[image_index]
            if count:
                query_classes[image_index, :count] = target.labels
                query_boxes[image_index, :count] = target.boxes
                padding_mask[image_index, :count] = True

        query_classes = query_classes.tile((1, 2 * num_groups))
        query_boxes = query_boxes.tile((1, 2 * num_groups, 1))
        padding_mask = padding_mask.tile((1, 2 * num_groups))

        negative_mask = torch.zeros((batch_size, max_targets * 2, 1), device=device)
        negative_mask[:, max_targets:] = 1
        negative_mask = negative_mask.tile((1, num_groups, 1))
        positive_mask = (1 - negative_mask).squeeze(-1) * padding_mask
        flat_positive_indices = torch.nonzero(positive_mask)[:, 1]
        positive_indices = tuple(
            torch.split(flat_positive_indices, [count * num_groups for count in target_counts])
        )

        actual_queries = max_targets * 2 * num_groups
        if cfg.label_noise_ratio > 0:
            label_noise = (
                torch.rand_like(query_classes, dtype=torch.float) < cfg.label_noise_ratio * 0.5
            )
            noisy_labels = torch.randint_like(
                label_noise, 0, self.num_classes, dtype=query_classes.dtype
            )
            query_classes = torch.where(label_noise & padding_mask, noisy_labels, query_classes)

        if cfg.box_noise_scale > 0:
            noisy_boxes = _box_cxcywh_to_xyxy(query_boxes)
            difference = torch.tile(query_boxes[..., 2:] * 0.5, (1, 1, 2)) * cfg.box_noise_scale
            random_sign = torch.randint_like(query_boxes, 0, 2) * 2.0 - 1.0
            random_part = torch.rand_like(query_boxes)
            random_part = (random_part + 1.0) * negative_mask + random_part * (1 - negative_mask)
            noisy_boxes += random_sign * random_part * difference
            query_boxes = _box_xyxy_to_cxcywh(noisy_boxes.clip(min=0.0, max=1.0))
            query_boxes = inverse_sigmoid(query_boxes)

        content = self.class_embed(query_classes)
        total_queries = actual_queries + num_queries
        attention_mask = torch.zeros(
            (total_queries, total_queries), dtype=torch.bool, device=device
        )
        attention_mask[actual_queries:, :actual_queries] = True
        group_size = max_targets * 2
        for group_index in range(num_groups):
            start = group_size * group_index
            end = start + group_size
            attention_mask[start:end, :start] = True
            attention_mask[start:end, end:actual_queries] = True

        return DenoisingBatch(
            queries=DenoisingQueries(content, query_boxes, attention_mask),
            metadata=DenoisingMetadata(
                positive_indices=positive_indices,
                num_groups=num_groups,
                num_queries=actual_queries,
            ),
        )
