# Copyright 2026 Clean RT-DETR contributors
"""Integration contracts for RT-DETRv3 branch loss normalization."""

import pytest
import torch
from test_model import _tiny

from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion
from rtdetrv3_recipe.criterion import RTDETRv3Criterion


def _target() -> list[DetectionTarget]:
    """Build the target for branch loss comparisons."""
    return [
        DetectionTarget(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        )
    ]


def test_invalid_criterion_configuration_is_rejected() -> None:
    """Reject invalid target multiplicity and branch weights."""
    with pytest.raises(ValueError, match="repeats"):
        RTDETRv3Criterion(repeats=0)
    with pytest.raises(ValueError, match="Branch weights"):
        RTDETRv3Criterion(weight_o2o=-1)
    with pytest.raises(ValueError, match="Branch weights"):
        RTDETRv3Criterion(weight_o2o=0, weight_o2m=0, weight_cnn=0)


def test_o2o_normalization_and_o2m_repeated_targets_match_set_criterion() -> None:
    """Compare group averaging and repeated-target supervision to the baseline."""
    model = _tiny().eval()
    images = torch.randn(1, 3, 64, 64)
    targets = _target()
    model.train()
    actual_output = model(images, targets=targets)
    criterion = RTDETRv3Criterion(weight_cnn=0, repeats=2)
    losses = criterion(actual_output, targets)
    groups = (actual_output.main, *actual_output.sap)
    reference = SetCriterion(matcher=HungarianMatcher())
    for index, group in enumerate(groups):
        expected = reference(group, targets)
        for name, value in expected.items():
            torch.testing.assert_close(losses[f"{name}_o2o_{index}"], value / len(groups))
    repeated = [DetectionTarget(targets[0].labels.repeat(2), targets[0].boxes.repeat(2, 1))]
    expected_o2m = reference(actual_output.o2m, repeated)
    for name, value in expected_o2m.items():
        torch.testing.assert_close(losses[f"{name}_o2m"], value)
