# Copyright 2026 Clean RT-DETR contributors
"""CPU parity and contract tests for matching and training losses."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from training_model_fixtures import small_model
from training_oracle import criterion_oracle, matcher_oracle

from clean_rtdetr import DecoderOutput, DetectionOutput, ModelOutput
from clean_rtdetr_training import (
    CriterionConfig,
    DetectionTarget,
    HungarianMatcher,
    MatcherConfig,
    SetCriterion,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TWO_IMAGES = 2
TWO_MATCHES = 2


def _targets(
    dtype: torch.dtype, *, empty: bool = False, mixed: bool = False
) -> list[DetectionTarget]:
    first = DetectionTarget(
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([[0.25, 0.3, 0.2, 0.25], [0.7, 0.65, 0.15, 0.2]], dtype=dtype),
    )
    second = DetectionTarget(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([[0.5, 0.45, 0.3, 0.25]], dtype=dtype),
    )
    empty_target = DetectionTarget(
        torch.empty(0, dtype=torch.int64), torch.empty((0, 4), dtype=dtype)
    )
    if empty:
        return [empty_target, empty_target]
    return [first, empty_target if mixed else second]


def _predictions(dtype: torch.dtype, *, queries: int = 4) -> DetectionOutput:
    torch.manual_seed(31)
    logits = torch.randn(2, queries, 3, dtype=dtype, requires_grad=True)
    # Keep boxes valid cxcywh while retaining nontrivial gradients.
    boxes = torch.rand(2, queries, 4, dtype=dtype)
    boxes[..., 2:] = boxes[..., 2:] * 0.35 + 0.05
    boxes.requires_grad_()
    return DetectionOutput(logits, boxes)


def _unchecked_output(logits: torch.Tensor, boxes: torch.Tensor) -> DetectionOutput:
    """Construct a malformed contract object while bypassing jaxtyping checks."""
    output = object.__new__(DetectionOutput)
    object.__setattr__(output, "logits", logits)
    object.__setattr__(output, "boxes", boxes)
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matcher_indices_match_pinned_upstream(dtype: torch.dtype) -> None:
    """Compare assignment indices directly against the unchanged upstream matcher."""
    predictions = _predictions(dtype)
    targets = _targets(dtype)
    actual = HungarianMatcher()(predictions, targets)
    expected = matcher_oracle(predictions, targets)
    assert len(actual) == TWO_IMAGES
    for match, (expected_prediction, expected_target) in zip(actual, expected, strict=True):
        torch.testing.assert_close(match.prediction_indices, expected_prediction)
        torch.testing.assert_close(match.target_indices, expected_target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["normal", "mixed", "empty"])
def test_criterion_values_and_gradients_match_pinned_oracle(
    dtype: torch.dtype,
    mode: str,
) -> None:
    """Weighted losses and gradients agree with the extracted upstream criterion."""
    predictions = _predictions(dtype)
    targets = _targets(dtype, empty=mode == "empty", mixed=mode == "mixed")
    actual = SetCriterion(matcher=HungarianMatcher())(predictions, targets)
    expected = criterion_oracle(predictions, targets)
    assert set(actual) == {"loss_vfl", "loss_bbox", "loss_giou"}
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=2e-5, atol=2e-6)
    sum(actual.values()).backward()
    assert predictions.logits.grad is not None
    assert predictions.boxes.grad is not None
    assert torch.isfinite(predictions.logits.grad).all()
    assert torch.isfinite(predictions.boxes.grad).all()

    oracle_predictions = DetectionOutput(
        predictions.logits.detach().clone().requires_grad_(),
        predictions.boxes.detach().clone().requires_grad_(),
    )
    oracle_losses = criterion_oracle(oracle_predictions, targets)
    torch.stack(list(oracle_losses.values())).sum().backward()
    torch.testing.assert_close(predictions.logits.grad, oracle_predictions.logits.grad)
    torch.testing.assert_close(predictions.boxes.grad, oracle_predictions.boxes.grad)


def test_target_count_above_query_count_and_match_cost_detachment() -> None:
    """Matching remains one-to-one when targets exceed the available queries."""
    predictions = _predictions(torch.float32, queries=2)
    target = DetectionTarget(
        torch.tensor([0, 1, 2]),
        torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]),
    )
    targets = [target, DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))]
    matches = HungarianMatcher()(predictions, targets)
    expected_matches = matcher_oracle(predictions, targets)
    for match, (rows, columns) in zip(matches, expected_matches, strict=True):
        torch.testing.assert_close(match.prediction_indices, rows)
        torch.testing.assert_close(match.target_indices, columns)
    actual_losses = SetCriterion(matcher=HungarianMatcher())(predictions, targets)
    expected_losses = criterion_oracle(predictions, targets)
    for name, value in actual_losses.items():
        torch.testing.assert_close(value, expected_losses[name])
    assert matches[0].prediction_indices.numel() == TWO_MATCHES
    assert matches[0].target_indices.numel() == TWO_MATCHES
    assert all(
        index.grad_fn is None
        for match in matches
        for index in (match.prediction_indices, match.target_indices)
    )


def test_vfl_iou_quality_is_detached_from_box_gradients() -> None:
    """VFL quality targets do not route gradients through predicted boxes."""
    predictions = _predictions(torch.float32)
    targets = _targets(torch.float32)
    criterion = SetCriterion(
        CriterionConfig(weight_vfl=1.0, weight_bbox=0.0, weight_giou=0.0),
        matcher=HungarianMatcher(),
    )
    criterion(predictions, targets)["loss_vfl"].backward()
    assert predictions.boxes.grad is None
    assert predictions.logits.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_output_auxiliary_stages_are_independently_matched_and_named(
    dtype: torch.dtype,
) -> None:
    """Decoder intermediates and encoder outputs receive independently named losses."""
    torch.manual_seed(4)
    decoder = DecoderOutput(
        torch.randn(3, 2, 4, 3, dtype=dtype, requires_grad=True),
        (torch.rand(3, 2, 4, 4, dtype=dtype) * 0.5).requires_grad_(),
    )
    encoder = DetectionOutput(
        torch.randn(2, 4, 3, dtype=dtype, requires_grad=True),
        (torch.rand(2, 4, 4, dtype=dtype) * 0.5).requires_grad_(),
    )
    output = ModelOutput(decoder, encoder, None)
    targets = _targets(dtype)
    losses = SetCriterion(matcher=HungarianMatcher())(output, targets)
    reference = ModelOutput(
        DecoderOutput(
            decoder.logits.detach().clone().requires_grad_(),
            decoder.boxes.detach().clone().requires_grad_(),
        ),
        DetectionOutput(
            encoder.logits.detach().clone().requires_grad_(),
            encoder.boxes.detach().clone().requires_grad_(),
        ),
        None,
    )
    expected = criterion_oracle(reference, targets)
    assert set(losses) == {
        "loss_vfl",
        "loss_bbox",
        "loss_giou",
        "loss_vfl_aux_0",
        "loss_bbox_aux_0",
        "loss_giou_aux_0",
        "loss_vfl_aux_1",
        "loss_bbox_aux_1",
        "loss_giou_aux_1",
        "loss_vfl_aux_2",
        "loss_bbox_aux_2",
        "loss_giou_aux_2",
    }
    for name, value in losses.items():
        torch.testing.assert_close(value, expected[name])
    sum(losses.values()).backward()
    torch.stack(list(expected.values())).sum().backward()
    for actual_tensor, reference_tensor in zip(
        (decoder.logits, decoder.boxes, encoder.logits, encoder.boxes),
        (
            reference.decoder.logits,
            reference.decoder.boxes,
            reference.encoder_aux.logits,
            reference.encoder_aux.boxes,
        ),
        strict=True,
    ):
        assert actual_tensor.grad is not None
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad)


@pytest.mark.parametrize(
    ("make_config", "match"),
    [
        (lambda: MatcherConfig(cost_class=0.0, cost_bbox=0.0, cost_giou=0.0), "positive"),
        (lambda: MatcherConfig(alpha=2.0), "alpha"),
        (lambda: CriterionConfig(weight_vfl=-1.0), "weights"),
        (lambda: CriterionConfig(gamma=-1.0), "gamma"),
    ],
)
def test_configs_validate_representative_invalid_values(
    make_config: Callable[[], MatcherConfig | CriterionConfig], match: str
) -> None:
    """Matcher and criterion configurations reject invalid representative values."""
    with pytest.raises(ValueError, match=match):
        make_config()


def test_criterion_backward_reaches_model_stages() -> None:
    """Real matching losses connect decoder and encoder heads to image features."""
    torch.manual_seed(43)
    model = small_model().train()
    output = model(torch.randn(1, 3, 64, 64))
    target = DetectionTarget(torch.tensor([1]), torch.tensor([[0.5, 0.5, 0.3, 0.2]]))
    losses = SetCriterion(matcher=HungarianMatcher())(output, [target])
    sum(losses.values()).backward()
    for module in (
        model.backbone,
        model.encoder,
        model.selector.enc_score_head,
        model.selector.enc_bbox_head,
        model.decoder.dec_score_head,
        model.decoder.dec_bbox_head,
    ):
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )


def test_custom_loss_configuration_matches_upstream() -> None:
    """Nondefault loss weights and focal parameters affect the same upstream terms."""
    output = _predictions(torch.float64)
    targets = _targets(torch.float64)
    config = CriterionConfig(weight_vfl=2.0, weight_bbox=3.0, weight_giou=4.0, alpha=0.5, gamma=1.0)
    matcher = HungarianMatcher(MatcherConfig(alpha=0.3, gamma=1.5))
    actual = SetCriterion(config, matcher=matcher)(output, targets)
    expected = criterion_oracle(
        output,
        targets,
        weight_vfl=2.0,
        weight_bbox=3.0,
        weight_giou=4.0,
        alpha=0.5,
        gamma=1.0,
        matcher_alpha=0.3,
        matcher_gamma=1.5,
    )
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name])


def test_invalid_inputs_are_rejected_before_matching() -> None:
    """Nonfinite, out-of-range, and malformed predictions fail at the public boundary."""
    predictions = _predictions(torch.float32)
    targets = _targets(torch.float32)
    with pytest.raises(ValueError, match="finite"):
        SetCriterion(matcher=HungarianMatcher())(
            replace(predictions, logits=predictions.logits.detach().clone().fill_(float("nan"))),
            targets,
        )
    with pytest.raises(ValueError, match="labels"):
        SetCriterion(matcher=HungarianMatcher())(
            predictions, [replace(targets[0], labels=torch.tensor([3, 0])), targets[1]]
        )
    with pytest.raises(ValueError, match="shape"):
        SetCriterion(matcher=HungarianMatcher())(
            _unchecked_output(predictions.logits, predictions.boxes[..., :3]), targets
        )
