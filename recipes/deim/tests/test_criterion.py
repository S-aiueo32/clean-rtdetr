# Copyright 2026 Clean RT-DETR contributors
"""Exercise the public criterion contract with normal, auxiliary, and DN stages."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

from clean_rtdetr.contracts import (
    DecoderOutput,
    DetectionOutput,
    ModelOutput,
)
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion
from clean_rtdetr_training.configuration import compose_training_config
from clean_rtdetr_training.contracts import DenoisingMetadata, DenoisingOutput, TrainingOutput
from deim_recipe.criterion import MALCriterion


def _output() -> TrainingOutput:
    """Use two layers, two DN groups, and a second image with no objects."""

    def decoder(queries: int) -> DecoderOutput:
        return DecoderOutput(
            torch.zeros(2, 2, queries, 2, dtype=torch.float64, requires_grad=True),
            torch.full((2, 2, queries, 4), 0.4, dtype=torch.float64, requires_grad=True),
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


def test_all_stages_reuse_box_semantics_and_dn_normalization() -> None:
    """Nonzero box errors and twice as many DN slots retain the shared reductions."""
    output, targets = _output(), _targets()
    actual = MALCriterion(matcher=HungarianMatcher())(output, targets)
    baseline = SetCriterion(matcher=HungarianMatcher())(output, targets)
    suffixes = ("", "_aux_0", "_aux_1", "_dn_0", "_dn_1")
    assert set(actual) == {
        f"loss_{kind}{suffix}" for kind in ("mal", "bbox", "giou") for suffix in suffixes
    }
    for suffix in suffixes:
        assert actual[f"loss_bbox{suffix}"] > 0
        assert actual[f"loss_giou{suffix}"] > 0
        torch.testing.assert_close(actual[f"loss_bbox{suffix}"], baseline[f"loss_bbox{suffix}"])
        torch.testing.assert_close(actual[f"loss_giou{suffix}"], baseline[f"loss_giou{suffix}"])
        torch.testing.assert_close(actual[f"loss_mal{suffix}"], actual["loss_mal"])
    sum(actual.values()).backward()
    assert output.predictions.decoder.logits.grad is not None
    assert output.denoising is not None
    assert output.denoising.decoder.logits.grad is not None
    assert torch.isfinite(output.denoising.decoder.logits.grad).all()
    assert output.predictions.decoder.boxes.grad is not None
    assert output.predictions.decoder.boxes.grad.abs().sum() > 0


def test_empty_targets_produce_only_negative_classification() -> None:
    """No matches still train negative logits and yield zero box losses."""
    output = _output().final
    losses = MALCriterion(matcher=HungarianMatcher()).stage_losses(output, [_targets()[1]] * 2)
    assert losses["loss_mal"] > 0
    assert losses["loss_bbox"] == 0
    assert losses["loss_giou"] == 0


def test_invalid_dn_metadata_is_rejected() -> None:
    """Known DN positives must agree with canonical repeated target slots."""
    output = _output()
    assert output.denoising is not None
    bad = replace(
        output.denoising.metadata,
        positive_indices=(torch.tensor([1, 3]), torch.empty(0, dtype=torch.int64)),
    )
    output = replace(output, denoising=replace(output.denoising, metadata=bad))
    with pytest.raises(ValueError, match="positive indices"):
        MALCriterion(matcher=HungarianMatcher())(output, _targets())


def test_yaml_injects_local_criterion_and_checkpoint_changes() -> None:
    """The installed composition API imports only the caller-owned criterion."""
    config = compose_training_config("train", config_dir=Path(__file__).parents[1] / "configs")
    criterion = instantiate(config.module.criterion)
    assert isinstance(criterion, MALCriterion)
    assert not isinstance(criterion, SetCriterion)
    assert criterion.checkpoint_config()["config"] == {
        "gamma": 1.5,
        "weight_mal": 1.0,
        "weight_bbox": 5.0,
        "weight_giou": 2.0,
    }
    changed = MALCriterion(matcher=HungarianMatcher(), gamma=2.0)
    assert criterion.checkpoint_config() != changed.checkpoint_config()
    assert (
        criterion.checkpoint_config()
        == MALCriterion(matcher=HungarianMatcher()).checkpoint_config()
    )


def test_zero_iou_match_is_classified_as_a_positive_without_box_gradient() -> None:
    """Recipe MAL keeps the matched-class unit weight even when IoU is zero."""
    output = DetectionOutput(
        torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True),
        torch.tensor([[[0.1, 0.1, 0.1, 0.1]]], dtype=torch.float64, requires_grad=True),
    )
    targets = [
        DetectionTarget(
            torch.tensor([1]), torch.tensor([[0.8, 0.8, 0.1, 0.1]], dtype=torch.float64)
        )
    ]
    losses = MALCriterion(matcher=HungarianMatcher()).stage_losses(output, targets)
    expected = torch.log(torch.tensor(2.0, dtype=torch.float64)) * (1 + 0.5**1.5)
    torch.testing.assert_close(losses["loss_mal"], expected)
    losses["loss_mal"].backward()
    torch.testing.assert_close(
        output.logits.grad, torch.tensor([[[0.5**2.5, 0.5]]], dtype=torch.float64)
    )
    assert output.boxes.grad is None


@pytest.mark.parametrize("weight_name", ["weight_mal", "weight_bbox", "weight_giou"])
def test_recipe_weights_affect_only_their_loss_and_resume_config(weight_name: str) -> None:
    """Each recorded weight controls its own loss at every retained stage."""
    output, targets = _output(), _targets()
    original = MALCriterion(matcher=HungarianMatcher())
    weight = getattr(original.config, weight_name)
    changed = MALCriterion(matcher=HungarianMatcher(), **{weight_name: weight * 3})
    assert original.checkpoint_config() != changed.checkpoint_config()
    baseline = original(output, targets)
    losses = changed(output, targets)
    prefix = weight_name.replace("weight_", "loss_")
    for name, value in baseline.items():
        multiplier = 3 if name == prefix or name.startswith(prefix + "_") else 1
        torch.testing.assert_close(losses[name], multiplier * value)


def test_recipe_rejects_all_disabled_loss_terms() -> None:
    """Unused reference VFL defaults cannot disguise a zero MAL objective."""
    with pytest.raises(ValueError, match="at least one loss weight"):
        MALCriterion(matcher=HungarianMatcher(), weight_mal=0, weight_bbox=0, weight_giou=0)
