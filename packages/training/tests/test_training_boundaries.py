# Copyright 2026 Clean RT-DETR contributors
"""Explicit eager validation and dependency failure boundaries for training."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from clean_rtdetr.contracts import DecoderOutput, DetectionOutput, ModelOutput
from clean_rtdetr_training import (
    CriterionConfig,
    DetectionTarget,
    HungarianMatcher,
    MatcherConfig,
    SetCriterion,
    TrainingOutput,
)

if TYPE_CHECKING:
    from torch import Tensor

_EXPECTED_BBOX_LOSS = 5.0


def _batch() -> tuple[DetectionOutput, list[DetectionTarget]]:
    """Return one image with valid prediction and target tensors."""
    return (
        DetectionOutput(torch.zeros(1, 2, 3), torch.full((1, 2, 4), 0.5)),
        [DetectionTarget(torch.tensor([1]), torch.full((1, 4), 0.25))],
    )


@pytest.mark.parametrize(
    ("field", "tensor", "message"),
    [
        ("logits", torch.zeros(2, 3), "rank"),
        ("logits", torch.zeros(0, 2, 3), "nonempty"),
        ("boxes", torch.zeros(1, 3, 4), "shape"),
        ("logits", torch.zeros(1, 2, 3, dtype=torch.float16), "float32 or float64"),
        ("boxes", torch.ones(1, 2, 4, dtype=torch.float64), "dtype"),
        ("boxes", torch.full((1, 2, 4), float("inf")), "finite"),
        ("boxes", torch.full((1, 2, 4), -0.1), "normalized"),
        ("boxes", torch.full((1, 2, 4), 1.1), "normalized"),
        ("boxes", torch.empty(1, 2, 4, device="meta"), "device"),
    ],
)
def test_prediction_eager_validation(field: str, tensor: Tensor, message: str) -> None:
    """Eager guards work even when annotations were bypassed by a caller."""
    output, targets = _batch()
    object.__setattr__(output, field, tensor)
    with pytest.raises(ValueError, match=message):
        SetCriterion(matcher=HungarianMatcher()).stage_losses(output, targets)


@pytest.mark.parametrize(
    ("field", "tensor", "message"),
    [
        ("labels", torch.tensor([1.0]), "int64"),
        ("labels", torch.tensor([[1]]), "one-dimensional"),
        ("labels", torch.tensor([-1]), "class range"),
        ("labels", torch.tensor([3]), "class range"),
        ("labels", torch.tensor([0, 1]), "lengths"),
        ("boxes", torch.zeros(1, 3), "shape"),
        ("boxes", torch.zeros(1, 4, dtype=torch.float64), "dtype"),
        ("labels", torch.empty(1, dtype=torch.int64, device="meta"), "device"),
        ("boxes", torch.full((1, 4), float("nan")), "finite"),
        ("boxes", torch.full((1, 4), -0.1), "normalized"),
        ("boxes", torch.full((1, 4), 1.1), "normalized"),
        ("boxes", torch.tensor([[0.5, 0.5, 0.0, 0.1]]), "positive"),
    ],
)
def test_target_eager_validation(field: str, tensor: Tensor, message: str) -> None:
    """Malformed target tensors cannot reach the assignment solver."""
    output, targets = _batch()
    object.__setattr__(targets[0], field, tensor)
    with pytest.raises(ValueError, match=message):
        SetCriterion(matcher=HungarianMatcher()).stage_losses(output, targets)


def test_batch_target_count_must_match() -> None:
    """Neither empty nor extra target lists describe a nonempty prediction batch."""
    output, targets = _batch()
    for invalid in ([], targets + targets):
        with pytest.raises(ValueError, match="one target"):
            SetCriterion(matcher=HungarianMatcher()).stage_losses(output, invalid)


@pytest.mark.parametrize("field", ["weight_vfl", "weight_bbox", "weight_giou", "alpha", "gamma"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_loss_config_rejects_invalid_numbers(field: str, value: float) -> None:
    """Loss configuration rejects nonfinite and negative parameters."""
    with pytest.raises(ValueError, match=r"finite|nonnegative|alpha"):
        replace(CriterionConfig(), **{field: value})


@pytest.mark.parametrize("field", ["cost_class", "cost_bbox", "cost_giou", "alpha", "gamma"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_matcher_config_rejects_invalid_numbers(field: str, value: float) -> None:
    """Assignment configuration rejects nonfinite and negative parameters."""
    with pytest.raises(ValueError, match=r"finite|nonnegative|alpha"):
        replace(MatcherConfig(), **{field: value})


def test_loss_config_rejects_all_zero_weights_and_alpha_above_one() -> None:
    """A criterion must retain a loss term and a valid focal alpha."""
    with pytest.raises(ValueError, match="not all zero"):
        CriterionConfig(weight_vfl=0.0, weight_bbox=0.0, weight_giou=0.0)
    with pytest.raises(ValueError, match="alpha"):
        CriterionConfig(alpha=1.1)


@pytest.mark.parametrize("scope", ["stage", "training"])
def test_distributed_execution_uses_bounded_normalization(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    """Distributed stages use the reduced target count instead of a local count."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda count, **_kwargs: count.mul_(2))
    output, targets = _batch()
    criterion = SetCriterion(matcher=HungarianMatcher())
    if scope == "stage":
        losses = criterion.stage_losses(output, targets)
        assert losses["loss_bbox"] == _EXPECTED_BBOX_LOSS
        return
    model_output = ModelOutput(
        DecoderOutput(output.logits.unsqueeze(0), output.boxes.unsqueeze(0)), output, None
    )
    losses = criterion(TrainingOutput(model_output), targets)
    assert losses["loss_bbox"] == _EXPECTED_BBOX_LOSS


def test_encoder_classes_must_match_decoder() -> None:
    """Do not silently train auxiliary predictions on a different class vocabulary."""
    output, targets = _batch()
    model_output = ModelOutput(
        DecoderOutput(output.logits.unsqueeze(0), output.boxes.unsqueeze(0)),
        DetectionOutput(torch.zeros(1, 2, 4), output.boxes),
        None,
    )
    with pytest.raises(ValueError, match="Encoder predictions"):
        SetCriterion(matcher=HungarianMatcher())(TrainingOutput(model_output), targets)


def test_training_import_and_empty_targets_do_not_need_scipy() -> None:
    """A fresh interpreter can import training and process empty targets without SciPy."""
    script = """
import sys
import torch
sys.modules['scipy'] = None
sys.modules['scipy.optimize'] = None
from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion
output = DetectionOutput(torch.zeros(1, 2, 3, requires_grad=True),
                         torch.zeros(1, 2, 4, requires_grad=True))
empty = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
losses = SetCriterion(matcher=HungarianMatcher()).stage_losses(output, [empty])
sum(losses.values()).backward()
assert output.boxes.grad is not None
assert output.boxes.grad.count_nonzero() == 0
target = DetectionTarget(torch.tensor([1]), torch.full((1, 4), 0.25))
try:
    SetCriterion(matcher=HungarianMatcher()).stage_losses(output, [target])
except ImportError as error:
    assert 'clean-rtdetr-training' in str(error)
else:
    raise AssertionError('Expected optional dependency error')
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
