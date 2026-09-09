# Copyright 2026 Clean RT-DETR contributors
"""Shared supervision preserves explicit assignments, reductions and gradients."""

from dataclasses import asdict

import torch
from torch.nn import functional

from clean_rtdetr.contracts import DecoderOutput, DetectionOutput, ModelOutput
from clean_rtdetr_training import CriterionConfig, DetectionTarget, HungarianMatcher, SetCriterion
from clean_rtdetr_training.contracts import DenoisingMetadata, DenoisingOutput, TrainingOutput
from clean_rtdetr_training.losses import (
    gather_matched_pairs,
    iter_matched_stages,
    match_stage,
    matched_box_losses,
    matched_iou_targets,
)


def _output() -> TrainingOutput:
    """Use two matching layers, two DN groups and an empty second image."""

    def decoder(queries: int) -> DecoderOutput:
        return DecoderOutput(
            torch.zeros(2, 2, queries, 2, dtype=torch.float64, requires_grad=True),
            torch.full((2, 2, queries, 4), 0.5, dtype=torch.float64, requires_grad=True),
        )

    matching = decoder(2)
    return TrainingOutput(
        ModelOutput(matching, DetectionOutput(matching.logits[0], matching.boxes[0]), None),
        DenoisingOutput(
            decoder(4),
            DenoisingMetadata((torch.tensor([0, 2]), torch.empty(0, dtype=torch.int64)), 2, 4),
        ),
    )


def _targets() -> list[DetectionTarget]:
    return [
        DetectionTarget(torch.tensor([1]), torch.full((1, 4), 0.5, dtype=torch.float64)),
        DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4, dtype=torch.float64)),
    ]


def test_shared_stages_preserve_order_and_dn_normalization() -> None:
    """Standard assignments and known DN positives remain explicit to each criterion."""
    output, targets = _output(), _targets()
    stages = list(iter_matched_stages(output, targets, HungarianMatcher()))
    assert [stage.suffix for stage in stages] == ["", "_aux_0", "_aux_1", "_dn_0", "_dn_1"]
    assert [stage.normalizer for stage in stages] == [1, 1, 1, 2, 2]
    assert [sum(len(match.prediction_indices) for match in stage.matches) for stage in stages] == [
        1,
        1,
        1,
        2,
        2,
    ]
    losses = []
    for stage in stages:
        pairs = gather_matched_pairs(stage.output, targets, stage.matches)
        scores, mask = matched_iou_targets(stage.output.logits, pairs)
        assert not scores.requires_grad
        assert mask.dtype == torch.bool
        torch.testing.assert_close(scores, mask.to(scores.dtype))
        classification = (
            functional.binary_cross_entropy_with_logits(
                stage.output.logits, scores, reduction="sum"
            )
            / stage.normalizer
        )
        bbox, giou = matched_box_losses(pairs, stage.normalizer)
        assert bbox == giou == 0
        losses.append(classification + bbox + giou)
    for loss in losses:
        torch.testing.assert_close(loss, losses[0])
    torch.stack(losses).sum().backward()
    assert output.predictions.decoder.logits.grad is not None
    assert output.denoising is not None
    assert output.denoising.decoder.logits.grad is not None
    assert torch.isfinite(output.denoising.decoder.logits.grad).all()


def test_final_and_auxiliary_stages_match_independently() -> None:
    """Reordering predictions between decoder layers changes each local assignment."""
    logits = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
    boxes = torch.tensor(
        [
            [[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]],
            [[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]],
        ],
        dtype=torch.float64,
    )
    output = ModelOutput(DecoderOutput(logits, boxes), DetectionOutput(logits[0], boxes[0]), None)
    stages = list(iter_matched_stages(TrainingOutput(output), _targets()[:1], HungarianMatcher()))
    assert [stage.matches[0].prediction_indices.tolist() for stage in stages] == [[0], [1], [1]]


def test_empty_targets_keep_negative_targets_and_differentiable_zero_boxes() -> None:
    """An all-empty batch clamps normalization to one without inventing positives."""
    output = _output().final
    targets = [_targets()[1]] * 2
    stage = match_stage(output, targets, HungarianMatcher())
    pairs = gather_matched_pairs(stage.output, targets, stage.matches)
    scores, mask = matched_iou_targets(stage.output.logits, pairs)
    bbox, giou = matched_box_losses(pairs, stage.normalizer)
    assert stage.normalizer == 1
    assert not mask.any()
    assert not scores.any()
    assert bbox == giou == 0
    assert bbox.requires_grad
    assert giou.requires_grad


def test_zero_iou_keeps_positive_membership_and_detaches_classification_targets() -> None:
    """Classification sees zero-IoU matches without a box gradient through IoU targets."""
    output = DetectionOutput(
        torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True),
        torch.tensor([[[0.1, 0.1, 0.1, 0.1]]], dtype=torch.float64, requires_grad=True),
    )
    targets = [
        DetectionTarget(
            torch.tensor([1]), torch.tensor([[0.8, 0.8, 0.1, 0.1]], dtype=torch.float64)
        )
    ]
    stage = match_stage(output, targets, HungarianMatcher())
    pairs = gather_matched_pairs(output, targets, stage.matches)
    scores, mask = matched_iou_targets(output.logits, pairs)
    assert not scores.any()
    assert mask.tolist() == [[[False, True]]]
    functional.binary_cross_entropy_with_logits(output.logits, scores).backward()
    assert output.logits.grad is not None
    assert output.boxes.grad is None
    bbox, giou = matched_box_losses(pairs, stage.normalizer)
    (bbox + giou).backward()
    assert output.boxes.grad is not None
    assert output.boxes.grad.abs().sum() > 0


def test_reference_checkpoint_contains_only_reference_settings() -> None:
    """The standard criterion retains its established objective metadata."""
    matcher = HungarianMatcher()
    assert SetCriterion(matcher=matcher).checkpoint_config() == {
        "config": asdict(CriterionConfig()),
        "matcher": matcher.checkpoint_config(),
        "matcher_class": "clean_rtdetr_training.losses.matcher.HungarianMatcher",
    }
