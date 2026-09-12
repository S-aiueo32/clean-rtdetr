# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR matching losses, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed prediction/target boundaries and bounded distributed normalization.
The reference objective is Varifocal, L1, and GIoU. See third_party/RT-DETR/README.md
for the pinned source and attribution.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional

from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr_training.contracts import DetectionTarget, TrainingOutput
from clean_rtdetr_training.interfaces import Matcher, TrainingCriterion
from clean_rtdetr_training.losses.supervision import (
    MatchedPairs,
    MatchedStage,
    gather_matched_pairs,
    iter_matched_stages,
    match_stage,
    matched_box_losses,
    matched_iou_targets,
)


@dataclass(frozen=True)
class CriterionConfig:
    """Default Varifocal and box-loss coefficients for the pinned R50vd criterion."""

    weight_vfl: float = 1.0
    weight_bbox: float = 5.0
    weight_giou: float = 2.0
    alpha: float = 0.75
    gamma: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid weights and focal parameters before computing losses."""
        weights = (self.weight_vfl, self.weight_bbox, self.weight_giou)
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
            msg = "Loss weights must be finite, nonnegative, and not all zero"
            raise ValueError(msg)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            msg = "Varifocal alpha must be finite and in [0, 1]"
            raise ValueError(msg)
        if not math.isfinite(self.gamma) or self.gamma < 0:
            msg = "Varifocal gamma must be finite and nonnegative"
            raise ValueError(msg)


class SetCriterion(TrainingCriterion[TrainingOutput]):
    """Return the reference weighted VFL, L1 and GIoU objective.

    TrainingOutput supplies final and auxiliary stages to match independently,
    plus optional known DN assignments. Use stage_losses for a DetectionOutput
    on its own. Sum returned scalar values before ``backward``.
    Recipes with another classification equation
    implement their own criterion and may reuse the set-supervision helpers.
    In initialized distributed execution, target counts are summed across ranks,
    divided by world size, and clamped to one before DDP averages gradients.
    """

    def __init__(
        self,
        config: CriterionConfig | None = None,
        *,
        matcher: Matcher,
    ) -> None:
        """Register the reference objective settings and assignment module."""
        super().__init__()
        self.config = CriterionConfig() if config is None else config
        self.matcher = matcher

    def forward(
        self,
        output: TrainingOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Apply the objective to ordinary stages and, when present, known DN stages."""
        losses = {}
        for stage in iter_matched_stages(output, targets, self.matcher):
            losses.update(self._matched_losses(stage, targets))
        return losses

    def stage_losses(
        self, output: DetectionOutput, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Match and supervise one detection stage with the reference objective."""
        return self._matched_losses(match_stage(output, targets, self.matcher), targets)

    def _matched_losses(
        self, stage: MatchedStage, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute weighted terms using the stage's assignments and normalizer."""
        pairs = gather_matched_pairs(stage.output, targets, stage.matches)
        vfl = self._varifocal_loss(stage.output.logits, pairs, stage.normalizer)
        bbox, giou = matched_box_losses(pairs, stage.normalizer)
        return {
            f"loss_vfl{stage.suffix}": vfl * self.config.weight_vfl,
            f"loss_bbox{stage.suffix}": bbox * self.config.weight_bbox,
            f"loss_giou{stage.suffix}": giou * self.config.weight_giou,
        }

    def checkpoint_config(self) -> dict[str, object]:
        """Describe reference objective and matcher settings for checkpoint resumes."""
        matcher_type = type(self.matcher)
        return {
            "config": asdict(self.config),
            "matcher": self.matcher.checkpoint_config(),
            "matcher_class": f"{matcher_type.__module__}.{matcher_type.__qualname__}",
        }

    def _varifocal_loss(
        self,
        logits: Float[Tensor, "batch queries classes"],
        pairs: MatchedPairs,
        normalizer: float,
    ) -> Float[Tensor, ""]:
        """Build dense IoU targets and reduce VFL over all queries and classes.

        Symbols in the implementation correspond to::

            t = target         one-hot matched class; all zeros for unmatched queries
            p = probability    detached sigmoid(logits)
            q = ious           detached IoU for each matched pair
            y = target_scores  q at matched classes; zero elsewhere
            w = weight         alpha * p**gamma * (1 - t) + y
            loss = sum(w * BCEWithLogits(logits, y)) / normalizer

        VarifocalNet Eq. (2), https://arxiv.org/pdf/2008.13367, motivates the
        IoU-weighted BCE. This implementation selects positives with t rather
        than q > 0: matched zero-IoU classes receive zero weight. Detached y and
        w leave the VFL gradient only through BCE's logits input. Unmatched
        queries contribute negative-class loss even when there are no pairs.

        The sum covers batch, queries, and classes. Its implementation retains
        upstream's mean-over-queries followed by multiplication by query count.
        """
        target_scores, positive_mask = matched_iou_targets(logits, pairs)
        target = positive_mask.to(torch.int64)
        probability = logits.sigmoid().detach()
        weight = (
            self.config.alpha * probability.pow(self.config.gamma) * (1 - target) + target_scores
        )
        vfl = functional.binary_cross_entropy_with_logits(
            logits, target_scores, weight=weight, reduction="none"
        )
        return vfl.mean(1).sum() * logits.shape[1] / normalizer
