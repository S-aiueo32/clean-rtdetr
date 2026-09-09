# Copyright 2026 Clean RT-DETR contributors
"""Recipe-owned MAL objective using explicit standard set-supervision helpers."""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr_training import DetectionTarget, Matcher, TrainingCriterion
from clean_rtdetr_training.contracts import TrainingOutput
from clean_rtdetr_training.losses import (
    MatchedStage,
    gather_matched_pairs,
    iter_matched_stages,
    match_stage,
    matched_box_losses,
    matched_iou_targets,
)

from .loss import MatchabilityAwareLoss


@dataclass(frozen=True)
class MALConfig:
    """Only the equation and weights used by this recipe's objective."""

    gamma: float = 1.5
    weight_mal: float = 1.0
    weight_bbox: float = 5.0
    weight_giou: float = 2.0

    def __post_init__(self) -> None:
        """Reject a disabled objective or invalid exponent and multipliers."""
        values = (self.gamma, self.weight_mal, self.weight_bbox, self.weight_giou)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            msg = "MAL gamma and loss weights must be finite and nonnegative"
            raise ValueError(msg)
        if self.gamma == 0 or not any((self.weight_mal, self.weight_bbox, self.weight_giou)):
            msg = "MAL gamma must be positive and at least one loss weight must be nonzero"
            raise ValueError(msg)


class MALCriterion(TrainingCriterion[TrainingOutput]):
    """Own MAL configuration and compose standard assignments and box supervision."""

    def __init__(
        self,
        *,
        matcher: Matcher,
        gamma: float = 1.5,
        weight_mal: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ) -> None:
        """Configure the recipe equation and weighted classification/box losses."""
        super().__init__()
        self.config = MALConfig(gamma, weight_mal, weight_bbox, weight_giou)
        self.matcher = matcher
        self.classification = MatchabilityAwareLoss(gamma)

    def forward(
        self,
        output: TrainingOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Combine MAL with L1/GIoU at ordinary and canonical DN stages."""
        losses = {}
        for stage in iter_matched_stages(output, targets, self.matcher):
            losses.update(self._matched_losses(stage, targets))
        return losses

    def stage_losses(
        self,
        output: DetectionOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute MAL losses for one standalone detection stage."""
        return self._matched_losses(match_stage(output, targets, self.matcher), targets)

    def _matched_losses(
        self, stage: MatchedStage, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Apply the unchanged MAL equation to one assigned prediction stage."""
        pairs = gather_matched_pairs(stage.output, targets, stage.matches)
        scores, positive_mask = matched_iou_targets(stage.output.logits, pairs)
        mal = self.classification(stage.output.logits, scores, positive_mask, stage.normalizer)
        bbox, giou = matched_box_losses(pairs, stage.normalizer)
        return {
            f"loss_mal{stage.suffix}": mal * self.config.weight_mal,
            f"loss_bbox{stage.suffix}": bbox * self.config.weight_bbox,
            f"loss_giou{stage.suffix}": giou * self.config.weight_giou,
        }

    def checkpoint_config(self) -> dict[str, object]:
        """Record the active objective and assignment settings for compatible resumes."""
        matcher_type = type(self.matcher)
        return {
            "config": asdict(self.config),
            "matcher": self.matcher.checkpoint_config(),
            "matcher_class": f"{matcher_type.__module__}.{matcher_type.__qualname__}",
        }
