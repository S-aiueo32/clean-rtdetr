# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for normalized and raw prediction box validation."""

from typing import cast

import pytest
import torch
from jaxtyping import TypeCheckError

from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher
from clean_rtdetr_training.contracts import PredictionBoxRange, validate_detection

_BOX_COORDINATES = 4


def _target() -> list[DetectionTarget]:
    """Return a single valid normalized target."""
    return [DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.2]]))]


def _raw_output(boxes: torch.Tensor | None = None) -> DetectionOutput:
    """Return predictions whose first center lies outside the unit interval."""
    return DetectionOutput(
        torch.tensor([[[0.0], [2.0]]]),
        boxes
        if boxes is not None
        else torch.tensor([[[1.5, -2.0, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]]),
    )


def test_default_policy_rejects_raw_boxes() -> None:
    """The default matcher retains strict normalized prediction validation."""
    with pytest.raises(ValueError, match="normalized"):
        HungarianMatcher()(_raw_output(), _target())


def test_unbounded_policy_matches_expected_raw_prediction() -> None:
    """The unbounded policy accepts raw centers and preserves assignment costs."""
    matches = HungarianMatcher(prediction_box_range="unbounded")(_raw_output(), _target())[0]
    torch.testing.assert_close(matches.prediction_indices, torch.tensor([1]))
    torch.testing.assert_close(matches.target_indices, torch.tensor([0]))


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        (torch.tensor([[[0.0, 0.0, -0.1, 0.2]]]), "nonnegative"),
        (torch.tensor([[[0.0, 0.0, 0.1, float("nan")]]]), "finite"),
        (torch.tensor([[[0.0, 0.0, 0.1, float("inf")]]]), "finite"),
        (torch.zeros(1, 2, 3), "shape"),
        (torch.zeros(1, 1, 4, dtype=torch.float16), "float32 or float64"),
    ],
)
def test_unbounded_policy_rejects_malformed_predictions(boxes: torch.Tensor, message: str) -> None:
    """The unbounded policy still rejects malformed prediction tensors."""
    if boxes.shape[-1] == _BOX_COORDINATES:
        output = DetectionOutput(torch.zeros(boxes.shape[0], boxes.shape[1], 1), boxes)
    else:
        output = _raw_output()
        object.__setattr__(output, "boxes", boxes)
    with pytest.raises(ValueError, match=message):
        HungarianMatcher(prediction_box_range="unbounded")(output, _target())


def test_unbounded_policy_keeps_targets_normalized() -> None:
    """Prediction relaxation does not relax target box validation."""
    target = [DetectionTarget(torch.tensor([0]), torch.tensor([[1.1, 0.5, 0.2, 0.2]]))]
    with pytest.raises(ValueError, match="normalized"):
        HungarianMatcher(prediction_box_range="unbounded")(_raw_output(), target)


def test_checkpoint_config_records_only_nondefault_policy() -> None:
    """Checkpoint metadata records only a nondefault prediction policy."""
    assert "prediction_box_range" not in HungarianMatcher().checkpoint_config()
    assert (
        HungarianMatcher(prediction_box_range="unbounded").checkpoint_config()[
            "prediction_box_range"
        ]
        == "unbounded"
    )


def test_unknown_prediction_box_policy_is_rejected() -> None:
    """Unknown policies fail validation or their runtime type checks."""
    invalid_policy = cast("PredictionBoxRange", "invalid")
    with pytest.raises((ValueError, TypeCheckError)):
        validate_detection(_raw_output(), _target(), prediction_box_range=invalid_policy)
    with pytest.raises((ValueError, TypeCheckError)):
        HungarianMatcher(prediction_box_range=invalid_policy)
