# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR matching losses, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed prediction/target boundaries and an explicit single-process scope.
Only the baseline Varifocal, L1, and GIoU losses are implemented; see
third_party/RT-DETR/README.md for the pinned source and attribution.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.core.contracts import DetectionOutput, ModelOutput
from clean_rtdetr.training.contracts import DetectionTarget, validate_detection
from clean_rtdetr.training.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou
from clean_rtdetr.training.matcher import HungarianMatcher


@dataclass(frozen=True)
class CriterionConfig:
    """Use the pinned R50vd weights and Varifocal parameters by default."""

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


class SetCriterion(nn.Module):
    """Return weighted matching losses for one stage or all retained model stages.

    ``ModelOutput`` adds independently matched decoder intermediate losses and
    the encoder auxiliary loss, in that order, using ``_aux_{index}`` suffixes.
    Values are already weighted: sum the returned values before ``backward``.
    This boundary accepts matching queries only, without denoising predictions.
    Distributed normalization and mixed-precision training are outside this API.
    """

    def __init__(
        self,
        config: CriterionConfig | None = None,
        *,
        matcher: HungarianMatcher | None = None,
    ) -> None:
        """Register the matcher and immutable baseline loss configuration."""
        super().__init__()
        self.config = CriterionConfig() if config is None else config
        self.matcher = HungarianMatcher() if matcher is None else matcher

    def forward(
        self,
        output: DetectionOutput | ModelOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Normalize every stage by the local target count, clamped to at least one."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            msg = "SetCriterion currently supports single-process normalization only"
            raise ValueError(msg)
        auxiliary: list[DetectionOutput] = []
        if isinstance(output, ModelOutput):
            output.decoder.validate()
            final = output.final
            validate_detection(output.encoder_aux, targets)
            if (
                output.encoder_aux.logits.shape[0] != final.logits.shape[0]
                or output.encoder_aux.logits.shape[-1] != final.logits.shape[-1]
                or output.encoder_aux.logits.dtype != final.logits.dtype
                or output.encoder_aux.logits.device != final.logits.device
            ):
                msg = "Encoder predictions must share decoder batch, classes, dtype, and device"
                raise ValueError(msg)
            auxiliary.extend(
                DetectionOutput(logits, boxes)
                for logits, boxes in zip(
                    output.decoder.logits[:-1], output.decoder.boxes[:-1], strict=True
                )
            )
            auxiliary.append(output.encoder_aux)
        else:
            final = output
        # The matcher validates each prediction/target boundary before arithmetic.
        losses = self._stage_losses(final, targets)
        for index, stage in enumerate(auxiliary):
            losses.update(
                (f"{name}_aux_{index}", value)
                for name, value in self._stage_losses(stage, targets).items()
            )
        return losses

    def _stage_losses(
        self,
        output: DetectionOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Recompute assignment and preserve upstream reductions and detach points."""
        matches = self.matcher(output, targets)
        normalizer = max(sum(target.labels.numel() for target in targets), 1)
        batch_indices = torch.cat(
            [
                torch.full_like(match.prediction_indices, index)
                for index, match in enumerate(matches)
            ]
        )
        prediction_indices = torch.cat([match.prediction_indices for match in matches])
        indices = (batch_indices, prediction_indices)
        target_boxes = torch.cat(
            [
                target.boxes[match.target_indices]
                for target, match in zip(targets, matches, strict=True)
            ]
        )
        labels = torch.cat(
            [
                target.labels[match.target_indices]
                for target, match in zip(targets, matches, strict=True)
            ]
        )
        matched_boxes = output.boxes[indices]
        predicted_xyxy = cxcywh_to_xyxy(matched_boxes)
        target_xyxy = cxcywh_to_xyxy(target_boxes)
        ious = torch.diag(box_iou(predicted_xyxy, target_xyxy)[0]).detach()

        logits = output.logits
        num_classes = logits.shape[-1]
        target_classes = torch.full(
            logits.shape[:2], num_classes, dtype=torch.int64, device=logits.device
        )
        target_classes[indices] = labels
        target = functional.one_hot(target_classes, num_classes + 1)[..., :-1]
        target_scores = torch.zeros_like(target_classes, dtype=logits.dtype)
        target_scores[indices] = ious.to(logits.dtype)
        target_scores = target_scores.unsqueeze(-1) * target
        probability = logits.sigmoid().detach()
        weight = (
            self.config.alpha * probability.pow(self.config.gamma) * (1 - target) + target_scores
        )
        vfl = functional.binary_cross_entropy_with_logits(
            logits, target_scores, weight=weight, reduction="none"
        )
        vfl = vfl.mean(1).sum() * logits.shape[1] / normalizer
        bbox = functional.l1_loss(matched_boxes, target_boxes, reduction="none").sum() / normalizer
        giou = (1 - torch.diag(generalized_box_iou(predicted_xyxy, target_xyxy))).sum() / normalizer
        return {
            "loss_vfl": vfl * self.config.weight_vfl,
            "loss_bbox": bbox * self.config.weight_bbox,
            "loss_giou": giou * self.config.weight_giou,
        }
