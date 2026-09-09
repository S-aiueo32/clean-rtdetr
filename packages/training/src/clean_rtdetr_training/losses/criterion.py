# Copyright 2026 Clean RT-DETR contributors
"""RT-DETR matching losses, adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed prediction/target boundaries and an explicit single-process scope.
Only the baseline Varifocal, L1, and GIoU losses are implemented; see
third_party/RT-DETR/README.md for the pinned source and attribution.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float, Int64, Shaped
from torch import Tensor
from torch.nn import functional

from clean_rtdetr.contracts import (
    DecoderOutput,
    DenoisingMetadata,
    DenoisingOutput,
    DetectionOutput,
    ModelOutput,
)
from clean_rtdetr_training.contracts import DetectionTarget, MatchIndices, validate_detection
from clean_rtdetr_training.interfaces import Matcher, TrainingCriterion
from clean_rtdetr_training.losses.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou


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


@dataclass(frozen=True)
class _MatchedPairs:
    """Aligned predictions and targets shared by classification and regression.

    ``indices`` locates each pair in the dense [batch, queries] predictions.
    The cxcywh boxes serve L1; their xyxy counterparts serve IoU and GIoU.
    All box tensors retain their gradients until a loss explicitly detaches them.
    """

    indices: tuple[Int64[Tensor, " matches"], Int64[Tensor, " matches"]]
    labels: Int64[Tensor, " matches"]
    predicted_cxcywh: Float[Tensor, "matches 4"]
    target_cxcywh: Float[Tensor, "matches 4"]
    predicted_xyxy: Float[Tensor, "matches 4"]
    target_xyxy: Float[Tensor, "matches 4"]


class SetCriterion(TrainingCriterion[DetectionOutput | ModelOutput]):
    """Return weighted matching losses for one stage or all retained model stages.

    ``ModelOutput`` adds independently matched decoder intermediate losses and
    the encoder auxiliary loss, in that order, using ``_aux_{index}`` suffixes.
    Values are already weighted: sum the returned values before ``backward``.
    Separate denoising predictions use known target assignments and ``_dn_{index}`` keys.
    Distributed normalization and mixed-precision training are outside this API.
    """

    def __init__(
        self,
        config: CriterionConfig | None = None,
        *,
        matcher: Matcher,
    ) -> None:
        """Register the matcher and immutable baseline loss configuration."""
        super().__init__()
        self.config = CriterionConfig() if config is None else config
        self.matcher = matcher

    def forward(
        self,
        output: DetectionOutput | ModelOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute final, auxiliary, and optional DN losses.

        Every ordinary stage is matched independently and normalized by its
        local batch target count, clamped to one.  DN stages use the known
        repeated positive indices and normalize by that count times the number
        of DN groups.
        """
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            msg = "SetCriterion currently supports single-process normalization only"
            raise ValueError(msg)
        final, auxiliary = self._split_stages(output, targets)
        losses = self._stage_losses(final, targets)
        for index, stage in enumerate(auxiliary):
            losses.update(
                (f"{name}_aux_{index}", value)
                for name, value in self._stage_losses(stage, targets).items()
            )
        if isinstance(output, ModelOutput) and output.denoising is not None:
            losses.update(self._denoising_losses(output.denoising, output, targets))
        return losses

    def _split_stages(
        self,
        output: DetectionOutput | ModelOutput,
        targets: Sequence[DetectionTarget],
    ) -> tuple[DetectionOutput, list[DetectionOutput]]:
        """Validate a model output and return final plus auxiliary stages."""
        if not isinstance(output, ModelOutput):
            return output, []
        output.decoder.validate()
        final = output.final
        self._validate_encoder_auxiliary(output.encoder_aux, final, targets)
        auxiliary = [
            DetectionOutput(logits, boxes)
            for logits, boxes in zip(
                output.decoder.logits[:-1], output.decoder.boxes[:-1], strict=True
            )
        ]
        auxiliary.append(output.encoder_aux)
        return final, auxiliary

    @staticmethod
    def _validate_encoder_auxiliary(
        auxiliary: DetectionOutput,
        final: DetectionOutput,
        targets: Sequence[DetectionTarget],
    ) -> None:
        """Check encoder predictions against targets and the final decoder stage."""
        validate_detection(auxiliary, targets)
        if (
            auxiliary.logits.shape[0] != final.logits.shape[0]
            or auxiliary.logits.shape[-1] != final.logits.shape[-1]
            or auxiliary.logits.dtype != final.logits.dtype
            or auxiliary.logits.device != final.logits.device
        ):
            msg = "Encoder predictions must share decoder batch, classes, dtype, and device"
            raise ValueError(msg)

    def checkpoint_config(self) -> dict[str, object]:
        """Describe criterion and matcher settings required for checkpoint resumes."""
        matcher_type = type(self.matcher)
        return {
            "config": asdict(self.config),
            "matcher": self.matcher.checkpoint_config(),
            "matcher_class": f"{matcher_type.__module__}.{matcher_type.__qualname__}",
        }

    def _denoising_losses(
        self,
        denoising: DenoisingOutput,
        output: ModelOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Supervise positive DN slots and classify all other slots as negatives."""
        decoded, metadata = denoising.decoder, denoising.metadata
        reference = output.decoder.logits
        matches = self._build_denoising_matches(denoising, targets, reference)
        normalizer = max(sum(target.labels.numel() for target in targets), 1) * metadata.num_groups
        losses = {}
        for index, (logits, boxes) in enumerate(zip(decoded.logits, decoded.boxes, strict=True)):
            stage = DetectionOutput(logits, boxes)
            validate_detection(stage, targets)
            losses.update(
                (f"{name}_dn_{index}", value)
                for name, value in self._matched_losses(stage, targets, matches, normalizer).items()
            )
        return losses

    def _build_denoising_matches(
        self,
        denoising: DenoisingOutput,
        targets: Sequence[DetectionTarget],
        reference: Float[Tensor, "layers batch queries classes"],
    ) -> list[MatchIndices]:
        """Build known target assignments after validating DN predictions and metadata."""
        decoded, metadata = denoising.decoder, denoising.metadata
        self._validate_denoising_predictions(decoded, reference)
        self._validate_denoising_metadata(metadata, decoded, targets)
        maximum = max(target.labels.numel() for target in targets)
        matches = []
        for target, positive in zip(targets, metadata.positive_indices, strict=True):
            target_indices = torch.arange(target.labels.numel(), device=reference.device)
            expected = (
                torch.arange(metadata.num_groups, device=reference.device)[:, None] * (2 * maximum)
                + target_indices[None, :]
            ).flatten()
            self._validate_positive_indices(positive, expected)
            matches.append(MatchIndices(positive, target_indices.repeat(metadata.num_groups)))
        return matches

    @staticmethod
    def _validate_denoising_predictions(
        decoded: DecoderOutput,
        reference: Float[Tensor, "layers batch queries classes"],
    ) -> None:
        """Check DN prediction layout before comparing it with matching predictions."""
        decoded.validate()
        if (
            decoded.logits.shape[:2] != reference.shape[:2]
            or decoded.logits.shape[-1] != reference.shape[-1]
            or decoded.logits.dtype != reference.dtype
            or decoded.logits.device != reference.device
        ):
            msg = (
                "Denoising predictions must share matching layers, batch, classes, "
                "dtype, and device"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_denoising_metadata(
        metadata: DenoisingMetadata,
        decoded: DecoderOutput,
        targets: Sequence[DetectionTarget],
    ) -> None:
        """Check group counts, per-image metadata, and the padded positive/negative layout."""
        if (
            type(metadata.num_groups) is not int
            or metadata.num_groups <= 0
            or type(metadata.num_queries) is not int
            or metadata.num_queries != decoded.logits.shape[2]
            or len(metadata.positive_indices) != len(targets)
        ):
            msg = "Invalid denoising group count, query count, or batch metadata"
            raise ValueError(msg)
        maximum = max(target.labels.numel() for target in targets)
        if maximum == 0 or metadata.num_queries != 2 * maximum * metadata.num_groups:
            msg = "Denoising query count must cover positive and negative target groups"
            raise ValueError(msg)

    @staticmethod
    def _validate_positive_indices(
        positive: Shaped[Tensor, "..."],
        expected: Int64[Tensor, " _positives"],
    ) -> None:
        """Check one image's positive DN slots against its repeated target positions."""
        if (
            positive.dtype != torch.int64
            or positive.device != expected.device
            or positive.ndim != 1
            or not torch.equal(positive, expected)
        ):
            msg = "Denoising positive indices must match repeated target slots"
            raise ValueError(msg)

    def _stage_losses(
        self,
        output: DetectionOutput,
        targets: Sequence[DetectionTarget],
    ) -> dict[str, Float[Tensor, ""]]:
        """Recompute assignment and preserve upstream reductions and detach points."""
        matches = self.matcher(output, targets)
        normalizer = max(sum(target.labels.numel() for target in targets), 1)
        return self._matched_losses(output, targets, matches, normalizer)

    def _matched_losses(
        self,
        output: DetectionOutput,
        targets: Sequence[DetectionTarget],
        matches: Sequence[MatchIndices],
        normalizer: int,
    ) -> dict[str, Float[Tensor, ""]]:
        """Apply the shared VFL, L1, and GIoU reductions to fixed assignments.

        Hungarian and DN assignments use the same loss path. The caller provides
        the clamped batch target count as ``normalizer``, multiplied by the group
        count for DN. Configured weights are applied here, after each reduction.
        """
        pairs = self._gather_matched_pairs(output, targets, matches)
        vfl = self._varifocal_loss(output.logits, pairs, normalizer)
        bbox, giou = self._regression_losses(pairs, normalizer)
        return {
            "loss_vfl": vfl * self.config.weight_vfl,
            "loss_bbox": bbox * self.config.weight_bbox,
            "loss_giou": giou * self.config.weight_giou,
        }

    @staticmethod
    def _gather_matched_pairs(
        output: DetectionOutput,
        targets: Sequence[DetectionTarget],
        matches: Sequence[MatchIndices],
    ) -> _MatchedPairs:
        """Gather aligned pairs across the batch without changing assignment order.

        Dense indices, target labels, and both box coordinate representations
        share the same leading pair dimension. Images with no matches contribute
        empty slices. Prediction boxes keep their autograd connection to output.
        """
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
        return _MatchedPairs(
            indices=indices,
            labels=labels,
            predicted_cxcywh=matched_boxes,
            target_cxcywh=target_boxes,
            predicted_xyxy=predicted_xyxy,
            target_xyxy=target_xyxy,
        )

    def _varifocal_loss(
        self,
        logits: Float[Tensor, "batch queries classes"],
        pairs: _MatchedPairs,
        normalizer: int,
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
        ious = torch.diag(box_iou(pairs.predicted_xyxy, pairs.target_xyxy)[0]).detach()
        num_classes = logits.shape[-1]
        target_classes = torch.full(
            logits.shape[:2], num_classes, dtype=torch.int64, device=logits.device
        )
        target_classes[pairs.indices] = pairs.labels
        target = functional.one_hot(target_classes, num_classes + 1)[..., :-1]
        target_scores = torch.zeros_like(target_classes, dtype=logits.dtype)
        target_scores[pairs.indices] = ious.to(logits.dtype)
        target_scores = target_scores.unsqueeze(-1) * target
        probability = logits.sigmoid().detach()
        weight = (
            self.config.alpha * probability.pow(self.config.gamma) * (1 - target) + target_scores
        )
        vfl = functional.binary_cross_entropy_with_logits(
            logits, target_scores, weight=weight, reduction="none"
        )
        return vfl.mean(1).sum() * logits.shape[1] / normalizer

    @staticmethod
    def _regression_losses(
        pairs: _MatchedPairs,
        normalizer: int,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """Return unweighted L1 and GIoU losses over matched pairs, in that order.

        L1 sums absolute cxcywh differences over pairs and coordinates. GIoU
        sums ``1 - GIoU`` over aligned xyxy pairs. Each sum is divided by the
        caller's positive normalizer. Empty pairs produce zero losses; otherwise
        both losses differentiate the gathered prediction boxes.
        """
        bbox = (
            functional.l1_loss(pairs.predicted_cxcywh, pairs.target_cxcywh, reduction="none").sum()
            / normalizer
        )
        giou = (
            1 - torch.diag(generalized_box_iou(pairs.predicted_xyxy, pairs.target_xyxy))
        ).sum() / normalizer
        return bbox, giou
