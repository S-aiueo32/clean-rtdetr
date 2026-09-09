# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the recipe-local CNN auxiliary branch."""

import math

import pytest
import torch

from clean_rtdetr import DetectionOutput, FeatureSpec, MultiScaleFeatures
from clean_rtdetr_training import DetectionTarget
from rtdetrv3_recipe.cnn import CNNAuxiliaryHead, CNNCriterion, CNNOutput


@pytest.mark.parametrize("locations", [1, 3])
def test_exact_quality_normalization_and_two_class_gradients(locations: int) -> None:
    """Perfect repeated positives preserve the analytic VFL and uniform-bin DFL."""
    logits = torch.zeros(1, locations, 2, requires_grad=True)
    distribution = torch.zeros(1, locations, 4, 3, requires_grad=True)
    box = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]]).expand(1, locations, 4)
    output = CNNOutput(
        DetectionOutput(logits, box),
        distribution,
        torch.full((locations, 2), 0.5),
        torch.full((locations, 2), 0.2),
    )
    target = DetectionTarget(torch.tensor([0]), box[0, :1])
    losses = CNNCriterion()(output, [target])
    torch.testing.assert_close(losses["loss_vfl"], torch.tensor(math.log(2) * (1 + 0.75 / 4)))
    torch.testing.assert_close(losses["loss_iou"], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(losses["loss_dfl"], torch.tensor(math.log(3) * 0.5))
    sum(losses.values()).backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[..., 0], torch.full((1, locations), -0.5 / locations))
    torch.testing.assert_close(logits.grad[..., 1], torch.full((1, locations), 0.09375 / locations))


def _head_output(batch: int = 1) -> tuple[CNNAuxiliaryHead, CNNOutput]:
    specs = (FeatureSpec(4, 8), FeatureSpec(8, 16))
    head = CNNAuxiliaryHead(specs, num_classes=2, reg_max=4)
    features = MultiScaleFeatures((torch.randn(batch, 4, 4, 4), torch.randn(batch, 8, 2, 2)), specs)
    return head, head(features)


def _target(*boxes: list[float], labels: list[int] | None = None) -> DetectionTarget:
    values = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return DetectionTarget(
        torch.tensor(labels if labels is not None else [0] * len(values), dtype=torch.int64), values
    )


def test_head_returns_flattened_dense_geometry_and_checkpoint_settings() -> None:
    """Validate dense flattening, normalized geometry, and metadata."""
    head, output = _head_output()
    assert output.detection.logits.shape == (1, 20, 2)
    assert output.distribution_logits.shape == (1, 20, 4, 5)
    assert output.anchor_points.shape == output.stride_scales.shape == (20, 2)
    assert bool(((output.detection.boxes >= 0) & (output.detection.boxes <= 1)).all())
    assert head.checkpoint_config()["reg_max"] == 4  # noqa: PLR2004 - configured test value


def test_assignment_has_multiple_positives_and_unique_conflict_resolution() -> None:
    """Ensure TAL produces several positives and one target per location."""
    _, output = _head_output()
    criterion = CNNCriterion(topk=5)
    target = _target([0.25, 0.25, 0.5, 0.5], [0.3, 0.3, 0.5, 0.5], labels=[0, 1])
    assignment = criterion._assign(output, target)  # noqa: SLF001 - assignment contract test
    assert assignment.indices.numel() > 1
    assert assignment.indices.unique().numel() == assignment.indices.numel()
    assert bool((assignment.quality >= 0).all())


def test_two_ground_truths_with_one_class_use_class_labels_for_scatter() -> None:
    """Multiple same-class targets remain valid dense assignments."""
    _, output = _head_output()
    target = _target([0.2, 0.2, 0.3, 0.3], [0.7, 0.7, 0.3, 0.3])
    losses = CNNCriterion(topk=3)(output, [target])
    assert all(torch.isfinite(value) for value in losses.values())


def test_unmatched_class_is_supervised_as_a_negative() -> None:
    """A class absent from targets still receives the negative VFL gradient."""
    _, output = _head_output()
    output.detection.logits.retain_grad()
    losses = CNNCriterion()(output, [_target([0.25, 0.25, 0.5, 0.5])])
    losses["loss_vfl"].backward()
    assert output.detection.logits.grad is not None
    assert bool((output.detection.logits.grad[..., 1] > 0).all())


def test_empty_targets_are_finite_and_produce_no_regression_gradient() -> None:
    """Empty images retain finite negative classification and zero regression."""
    _, output = _head_output()
    output.detection.boxes.retain_grad()
    empty = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
    losses = CNNCriterion()(output, [empty])
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_iou"].backward()
    assert output.detection.boxes.grad is None


def test_all_losses_backpropagate_to_head_parameters() -> None:
    """All enabled loss terms preserve a finite gradient path to the head."""
    head, output = _head_output()
    target = _target([0.25, 0.25, 0.5, 0.5])
    losses = CNNCriterion()(output, [target])
    sum(losses.values()).backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
        if parameter.grad is not None
    )
