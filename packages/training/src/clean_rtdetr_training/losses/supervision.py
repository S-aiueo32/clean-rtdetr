# Copyright 2026 Clean RT-DETR contributors
"""Standard single-process set supervision shared by explicit training criteria.

Stage order, independent assignments, and canonical DN target repetition follow
RT-DETR. Recipes that change those policies should implement their own stage
supervision; these helpers do not dispatch loss implementations or own weights.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor
from torch.nn import functional

from clean_rtdetr.contracts import DecoderOutput, DetectionOutput, ModelOutput
from clean_rtdetr_training.contracts import (
    DenoisingMetadata,
    DenoisingOutput,
    DetectionTarget,
    MatchIndices,
    TrainingOutput,
    validate_detection,
)
from clean_rtdetr_training.interfaces import Matcher
from clean_rtdetr_training.losses.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou


@dataclass(frozen=True)
class MatchedStage:
    """One supervised stage, its assignment, key suffix and positive normalizer.

    Ordinary stages use the clamped local batch target count. DN stages multiply
    that count by the number of groups, and use known positive assignments.
    """

    suffix: str
    output: DetectionOutput
    matches: Sequence[MatchIndices]
    normalizer: int


@dataclass(frozen=True)
class MatchedPairs:
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


def iter_matched_stages(
    output: TrainingOutput,
    targets: Sequence[DetectionTarget],
    matcher: Matcher,
) -> Iterator[MatchedStage]:
    """Yield final, decoder/encoder auxiliary, then optional canonical DN stages.

    Each ordinary stage calls the supplied validating matcher independently.
    DN predictions and their known target assignments are validated here. Keys
    use an empty suffix for final, ``_aux_{index}`` for auxiliary predictions,
    and ``_dn_{index}`` for DN layers. The policy is single-process and uses
    normalized RT-DETR boxes; a different supervision policy belongs to its
    criterion rather than options on this helper.
    """
    normalizer = _target_normalizer(targets)
    final, auxiliary = _split_stages(output.predictions, targets)
    yield MatchedStage("", final, matcher(final, targets), normalizer)
    for index, stage in enumerate(auxiliary):
        yield MatchedStage(f"_aux_{index}", stage, matcher(stage, targets), normalizer)
    if output.denoising is not None:
        yield from _denoising_stages(output.denoising, output.predictions, targets, normalizer)


def match_stage(
    output: DetectionOutput,
    targets: Sequence[DetectionTarget],
    matcher: Matcher,
) -> MatchedStage:
    """Match one detection stage with a clamped, single-process target count."""
    normalizer = _target_normalizer(targets)
    return MatchedStage("", output, matcher(output, targets), normalizer)


def _target_normalizer(targets: Sequence[DetectionTarget]) -> int:
    """Reject distributed execution before using local target normalization."""
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        msg = "Set supervision currently supports single-process normalization only"
        raise ValueError(msg)
    return max(sum(target.labels.numel() for target in targets), 1)


def _denoising_stages(
    denoising: DenoisingOutput,
    predictions: ModelOutput,
    targets: Sequence[DetectionTarget],
    normalizer: int,
) -> Iterator[MatchedStage]:
    """Use known repeated target assignments for each DN decoder layer."""
    matches = _build_denoising_matches(denoising, targets, predictions.decoder.logits)
    normalizer *= denoising.metadata.num_groups
    decoded = denoising.decoder
    for index, (logits, boxes) in enumerate(zip(decoded.logits, decoded.boxes, strict=True)):
        stage = DetectionOutput(logits, boxes)
        validate_detection(stage, targets)
        yield MatchedStage(f"_dn_{index}", stage, matches, normalizer)


def _split_stages(
    output: ModelOutput,
    targets: Sequence[DetectionTarget],
) -> tuple[DetectionOutput, list[DetectionOutput]]:
    """Validate a model output and return final plus auxiliary stages."""
    output.decoder.validate()
    final = output.final
    _validate_encoder_auxiliary(output.encoder_aux, final, targets)
    auxiliary = [
        DetectionOutput(logits, boxes)
        for logits, boxes in zip(output.decoder.logits[:-1], output.decoder.boxes[:-1], strict=True)
    ]
    auxiliary.append(output.encoder_aux)
    return final, auxiliary


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


def _build_denoising_matches(
    denoising: DenoisingOutput,
    targets: Sequence[DetectionTarget],
    reference: Float[Tensor, "layers batch queries classes"],
) -> list[MatchIndices]:
    """Build known target assignments after validating DN predictions and metadata."""
    decoded, metadata = denoising.decoder, denoising.metadata
    _validate_denoising_predictions(decoded, reference)
    _validate_denoising_metadata(metadata, decoded, targets)
    maximum = max(target.labels.numel() for target in targets)
    matches = []
    for target, positive in zip(targets, metadata.positive_indices, strict=True):
        target_indices = torch.arange(target.labels.numel(), device=reference.device)
        expected = (
            torch.arange(metadata.num_groups, device=reference.device)[:, None] * (2 * maximum)
            + target_indices[None, :]
        ).flatten()
        _validate_positive_indices(positive, expected)
        matches.append(MatchIndices(positive, target_indices.repeat(metadata.num_groups)))
    return matches


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
        msg = "Denoising predictions must share matching layers, batch, classes, dtype, and device"
        raise ValueError(msg)


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


def gather_matched_pairs(
    output: DetectionOutput,
    targets: Sequence[DetectionTarget],
    matches: Sequence[MatchIndices],
) -> MatchedPairs:
    """Gather aligned pairs across the batch without changing assignment order.

    Dense indices, target labels, and both box coordinate representations
    share the same leading pair dimension. Images with no matches contribute
    empty slices. Prediction boxes keep their autograd connection to output.
    """
    batch_indices = torch.cat(
        [torch.full_like(match.prediction_indices, index) for index, match in enumerate(matches)]
    )
    prediction_indices = torch.cat([match.prediction_indices for match in matches])
    indices = (batch_indices, prediction_indices)
    target_boxes = torch.cat(
        [target.boxes[match.target_indices] for target, match in zip(targets, matches, strict=True)]
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
    return MatchedPairs(
        indices=indices,
        labels=labels,
        predicted_cxcywh=matched_boxes,
        target_cxcywh=target_boxes,
        predicted_xyxy=predicted_xyxy,
        target_xyxy=target_xyxy,
    )


def matched_iou_targets(
    logits: Float[Tensor, "batch queries classes"],
    pairs: MatchedPairs,
) -> tuple[Float[Tensor, "batch queries classes"], Bool[Tensor, "batch queries classes"]]:
    """Build detached IoU targets and retain zero-IoU positive membership."""
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
    return target_scores, target.bool()


def matched_box_losses(
    pairs: MatchedPairs,
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
