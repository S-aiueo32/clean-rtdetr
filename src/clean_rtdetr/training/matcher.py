# Copyright 2026 Clean RT-DETR contributors
"""Hungarian matching adapted from the pinned RT-DETR implementation.

The upstream implementation is vendored under ``third_party/RT-DETR`` and is
Apache-2.0 licensed; this module adds typed contracts and explicit validation.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from clean_rtdetr.core.contracts import DetectionOutput
from clean_rtdetr.training.contracts import DetectionTarget, MatchIndices, validate_detection
from clean_rtdetr.training.geometry import cxcywh_to_xyxy, generalized_box_iou
from clean_rtdetr.training.interfaces import Matcher


@dataclass(frozen=True)
class MatcherConfig:
    """Weights and focal-loss parameters for Hungarian matching."""

    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    alpha: float = 0.25
    gamma: float = 2.0

    def __post_init__(self) -> None:
        """Validate matcher weights and focal-loss parameters."""
        values = (self.cost_class, self.cost_bbox, self.cost_giou, self.alpha, self.gamma)
        if not all(math.isfinite(value) for value in values):
            msg = "Matcher parameters must be finite"
            raise ValueError(msg)
        if min(self.cost_class, self.cost_bbox, self.cost_giou) < 0:
            msg = "Matcher costs must be nonnegative"
            raise ValueError(msg)
        if self.cost_class == self.cost_bbox == self.cost_giou == 0:
            msg = "At least one matcher cost must be positive"
            raise ValueError(msg)
        if not 0 <= self.alpha <= 1:
            msg = "Matcher alpha must be in [0, 1]"
            raise ValueError(msg)
        if self.gamma < 0:
            msg = "Matcher gamma must be nonnegative"
            raise ValueError(msg)


class HungarianMatcher(Matcher):
    """Select a minimum-cost one-to-one assignment for each image."""

    def __init__(self, config: MatcherConfig | None = None) -> None:
        """Create a matcher with the supplied configuration."""
        super().__init__()
        self.config = config or MatcherConfig()

    def checkpoint_config(self) -> dict[str, object]:
        """Describe assignment costs and focal-loss settings for checkpoint resumes."""
        return asdict(self.config)

    @staticmethod
    def _solve(
        cost: Float[Tensor, " queries targets"],
    ) -> tuple[Int64[Tensor, " matches"], Int64[Tensor, " matches"]]:
        """Solve one CPU cost matrix, importing SciPy only when needed."""
        try:
            from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
        except ImportError as error:
            msg = (
                "HungarianMatcher requires scipy; install with `pip install clean-rtdetr[training]`"
            )
            raise ImportError(msg) from error
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(columns, dtype=torch.int64)

    @torch.no_grad()
    def forward(
        self, predictions: DetectionOutput, targets: Sequence[DetectionTarget]
    ) -> tuple[MatchIndices, ...]:
        """Return matched prediction and target indices, preserving prediction device."""
        validate_detection(predictions, targets)
        logits, boxes = predictions.logits, predictions.boxes
        results: list[MatchIndices] = []
        for image_logits, image_boxes, target in zip(logits, boxes, targets, strict=True):
            target_count = target.labels.shape[0]
            if target_count == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                results.append(MatchIndices(empty, empty.clone()))
                continue
            probabilities = image_logits.sigmoid()[:, target.labels]
            neg_cost = (
                (1 - self.config.alpha)
                * probabilities.pow(self.config.gamma)
                * (-(1 - probabilities + 1e-8).log())
            )
            pos_cost = (
                self.config.alpha
                * (1 - probabilities).pow(self.config.gamma)
                * (-(probabilities + 1e-8).log())
            )
            cost_class = pos_cost - neg_cost
            cost_bbox = torch.cdist(image_boxes, target.boxes, p=1)
            cost_giou = -generalized_box_iou(
                cxcywh_to_xyxy(image_boxes), cxcywh_to_xyxy(target.boxes)
            )
            cost = (
                self.config.cost_bbox * cost_bbox
                + self.config.cost_class * cost_class
                + self.config.cost_giou * cost_giou
            )
            prediction_indices, target_indices = self._solve(cost)
            results.append(
                MatchIndices(
                    prediction_indices.to(device=logits.device),
                    target_indices.to(device=logits.device),
                )
            )
        return tuple(results)
