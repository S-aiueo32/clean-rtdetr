# Copyright 2026 Clean RT-DETR contributors
"""Integration and upstream-parity checks for denoising training queries."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.core.contracts import (
    DecoderOutput,
    DenoisingMetadata,
    DenoisingOutput,
    DetectionOutput,
    ModelOutput,
)
from clean_rtdetr.models.decoder import DecoderPlan
from clean_rtdetr.training import DetectionTarget, HungarianMatcher, SetCriterion
from clean_rtdetr.training.denoising import DenoisingConfig, DenoisingQueryGenerator
from clean_rtdetr.training.model import DenoisingTrainingModel
from tests.test_model import baseline_model, small_model
from tests.training_oracle import criterion_oracle

if TYPE_CHECKING:
    from pathlib import Path


def _targets(
    *, empty: bool = False, mixed_empty: bool = False, dtype: torch.dtype = torch.float32
) -> list[DetectionTarget]:
    no_objects = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4, dtype=dtype))
    if empty:
        return [no_objects, no_objects]
    return [
        DetectionTarget(
            torch.tensor([0, 2]),
            torch.tensor([[0.25, 0.3, 0.2, 0.25], [0.7, 0.65, 0.15, 0.2]], dtype=dtype),
        ),
        no_objects
        if mixed_empty
        else DetectionTarget(
            torch.tensor([1]), torch.tensor([[0.5, 0.45, 0.3, 0.25]], dtype=dtype)
        ),
    ]


def _synthetic_output(
    dtype: torch.dtype = torch.float64, *, mixed_empty: bool = False
) -> ModelOutput:
    torch.manual_seed(71)
    matching = DecoderOutput(
        torch.randn(2, 2, 3, 3, dtype=dtype, requires_grad=True),
        torch.rand(2, 2, 3, 4, dtype=dtype, requires_grad=True),
    )
    encoder = DetectionOutput(
        torch.randn(2, 3, 3, dtype=dtype, requires_grad=True),
        torch.rand(2, 3, 4, dtype=dtype, requires_grad=True),
    )
    denoising = DecoderOutput(
        torch.randn(2, 2, 8, 3, dtype=dtype, requires_grad=True),
        torch.rand(2, 2, 8, 4, dtype=dtype, requires_grad=True),
    )
    metadata = DenoisingMetadata(
        positive_indices=(
            torch.tensor([0, 1, 4, 5]),
            torch.empty(0, dtype=torch.int64) if mixed_empty else torch.tensor([0, 4]),
        ),
        num_groups=2,
        num_queries=8,
    )
    return ModelOutput(matching, encoder, None, DenoisingOutput(denoising, metadata))


def _dn_model(config: DenoisingConfig | None = None) -> DenoisingTrainingModel:
    model = small_model()
    generator = DenoisingQueryGenerator(
        num_classes=3,
        hidden_dim=8,
        config=DenoisingConfig(num_denoising=4) if config is None else config,
    )
    return DenoisingTrainingModel(model, generator)


def _official_wrapper_state() -> tuple[DenoisingTrainingModel, dict[str, torch.Tensor]]:
    detector = baseline_model()
    wrapper = DenoisingTrainingModel(detector, DenoisingQueryGenerator(80, 256))
    state: dict[str, torch.Tensor] = {}
    for key, value in detector.state_dict().items():
        if key.startswith(("backbone.", "encoder.")):
            source = key
        elif key.startswith("projector."):
            source = "decoder.input_proj." + key.removeprefix("projector.input_proj.")
        elif key.startswith("selector."):
            source = "decoder." + key.removeprefix("selector.")
        elif key.startswith("decoder.layers."):
            source = "decoder.decoder." + key.removeprefix("decoder.")
        else:
            source = key
        state[source] = value.detach().clone()
    state["decoder.denoising_class_embed.weight"] = torch.randn_like(
        wrapper.denoising.class_embed.weight
    )
    return wrapper, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mixed_empty", [False, True])
def test_dn_criterion_values_and_gradients_match_pinned_upstream(
    dtype: torch.dtype, *, mixed_empty: bool
) -> None:
    """Every retained DN layer uses upstream assignments, names, and normalization."""
    output = _synthetic_output(dtype, mixed_empty=mixed_empty)
    targets = _targets(dtype=dtype, mixed_empty=mixed_empty)
    actual = SetCriterion(matcher=HungarianMatcher())(output, targets)

    reference = _synthetic_output(dtype, mixed_empty=mixed_empty)
    expected = criterion_oracle(reference, targets)
    dn_names = {f"loss_{kind}_dn_{layer}" for layer in range(2) for kind in ("vfl", "bbox", "giou")}
    assert dn_names <= actual.keys()
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6)

    torch.stack(list(actual.values())).sum().backward()
    torch.stack(list(expected.values())).sum().backward()
    assert output.denoising is not None
    assert reference.denoising is not None
    for actual_tensor, expected_tensor in zip(
        (
            output.decoder.logits,
            output.decoder.boxes,
            output.encoder_aux.logits,
            output.encoder_aux.boxes,
            output.denoising.decoder.logits,
            output.denoising.decoder.boxes,
        ),
        (
            reference.decoder.logits,
            reference.decoder.boxes,
            reference.encoder_aux.logits,
            reference.encoder_aux.boxes,
            reference.denoising.decoder.logits,
            reference.denoising.decoder.boxes,
        ),
        strict=True,
    ):
        assert actual_tensor.grad is not None
        torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, rtol=2e-5, atol=2e-6)


def test_model_splits_dn_predictions_and_preserves_matching_outputs() -> None:
    """Random DN queries remain masked from the matching query sequence."""
    model = _dn_model().train()
    images = torch.randn(2, 3, 64, 64)
    targets = _targets()
    torch.manual_seed(101)
    first = model(images, targets=targets)
    torch.manual_seed(202)
    second = model(images, targets=targets)
    assert first.denoising is not None
    assert first.decoder.logits.shape == (2, 2, 2, 3)
    assert first.denoising.decoder.logits.shape == (2, 2, 8, 3)
    assert not torch.equal(first.denoising.decoder.logits, second.denoising.decoder.logits)
    torch.testing.assert_close(first.decoder.logits, second.decoder.logits)
    torch.testing.assert_close(first.decoder.boxes, second.decoder.boxes)


def test_mixed_targets_dn_losses_reach_class_embedding_and_obey_plan() -> None:
    """DN remains trainable with uneven target counts and a decoder layer prefix."""
    model = _dn_model().train()
    output = model(
        torch.randn(2, 3, 64, 64),
        targets=_targets(),
        plan=DecoderPlan(num_layers=1),
    )
    assert output.denoising is not None
    assert output.denoising.decoder.logits.shape[:3] == (1, 2, 8)
    losses = SetCriterion(matcher=HungarianMatcher())(output, _targets())
    assert {name for name in losses if "_dn_" in name} == {
        "loss_vfl_dn_0",
        "loss_bbox_dn_0",
        "loss_giou_dn_0",
    }
    torch.stack([value for name, value in losses.items() if "_dn_" in name]).sum().backward()
    gradient = model.denoising.class_embed.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_all_empty_and_disabled_dn_batches_have_no_dn_predictions() -> None:
    """Empty batches and an explicitly disabled configuration keep a stable schema."""
    images = torch.randn(2, 3, 64, 64)
    empty_targets = _targets(empty=True)
    assert _dn_model().train()(images, targets=empty_targets).denoising is None
    disabled = _dn_model(DenoisingConfig(num_denoising=0)).train()
    assert disabled(images, targets=_targets()).denoising is None
    with pytest.raises(ValueError, match="target"):
        disabled(images)


def test_eval_ignores_configured_dn_and_rejects_targets() -> None:
    """A configured generator does not alter target-free inference."""
    model = _dn_model().eval()
    images = torch.randn(2, 3, 64, 64)
    rng_before = torch.random.get_rng_state()
    output = model(images)
    rng_after = torch.random.get_rng_state()
    core_output = model.model(images)
    assert output.denoising is None
    assert torch.equal(rng_before, rng_after)
    torch.testing.assert_close(output.decoder.logits, core_output.decoder.logits)
    torch.testing.assert_close(output.decoder.boxes, core_output.decoder.boxes)
    torch.testing.assert_close(output.encoder_aux.logits, core_output.encoder_aux.logits)
    torch.testing.assert_close(output.encoder_aux.boxes, core_output.encoder_aux.boxes)
    with pytest.raises(ValueError, match="training mode"):
        model(images, targets=_targets())


def test_targets_require_dn_and_match_image_batch() -> None:
    """The model rejects ambiguous target use before running its image stages."""
    images = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError, match="target"):
        _dn_model().train()(images)
    with pytest.raises(ValueError, match="one target"):
        _dn_model().train()(images, targets=_targets()[:1])


@pytest.mark.parametrize(
    "metadata",
    [
        DenoisingMetadata((torch.tensor([0, 1, 4, 5]),), 2, 8),
        DenoisingMetadata((torch.tensor([0, 1, 4, 6]), torch.tensor([0, 4])), 2, 8),
        DenoisingMetadata((torch.tensor([0, 1, 4, 5]), torch.tensor([0, 4])), 0, 8),
        DenoisingMetadata((torch.tensor([0, 1, 4, 5]), torch.tensor([0, 4])), 2, 7),
    ],
)
def test_criterion_rejects_malformed_dn_metadata(metadata: DenoisingMetadata) -> None:
    """Metadata must describe the canonical positive slots and complete DN sequence."""
    output = _synthetic_output()
    assert output.denoising is not None
    malformed = replace(output, denoising=replace(output.denoising, metadata=metadata))
    with pytest.raises(ValueError, match=r"Denoising|denoising"):
        SetCriterion(matcher=HungarianMatcher())(malformed, _targets(dtype=torch.float64))


def test_criterion_rejects_dn_matching_dimension_mismatch() -> None:
    """DN predictions share matching layer, batch, class, dtype, and device dimensions."""
    output = _synthetic_output()
    assert output.denoising is not None
    bad_decoder = DecoderOutput(
        output.denoising.decoder.logits[:, :, :, :2],
        output.denoising.decoder.boxes,
    )
    malformed = replace(output, denoising=replace(output.denoising, decoder=bad_decoder))
    with pytest.raises(ValueError, match="share matching"):
        SetCriterion(matcher=HungarianMatcher())(malformed, _targets(dtype=torch.float64))


def test_dn_embedding_round_trips_in_training_wrapper_state_dict() -> None:
    """The separately composed class embedding is registered as checkpoint state."""
    configured = _dn_model()
    key = "denoising.class_embed.weight"
    assert key in configured.state_dict()
    clone = _dn_model()
    clone.load_state_dict(configured.state_dict())
    torch.testing.assert_close(clone.state_dict()[key], configured.state_dict()[key])


def test_official_loader_maps_dn_embedding_into_training_wrapper(tmp_path: Path) -> None:
    """Official DN weights load into the wrapper-owned embedding."""
    wrapper, state = _official_wrapper_state()
    marker = state["decoder.denoising_class_embed.weight"]
    path = tmp_path / "dn.pth"
    torch.save({"model": state}, path)
    report = load_official_model_weights(
        path,
        model=wrapper,
        source="model",
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert report.key_mapping["decoder.denoising_class_embed.weight"] == (
        "denoising.class_embed.weight"
    )
    torch.testing.assert_close(wrapper.denoising.class_embed.weight, marker)


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_invalid_official_dn_embedding_is_atomic(tmp_path: Path, failure: str) -> None:
    """DN validation failures occur before any detector or embedding parameter changes."""
    wrapper, state = _official_wrapper_state()
    detector_key = "backbone.conv1.conv1_1.conv.weight"
    state[detector_key].fill_(0.25)
    dn_key = "decoder.denoising_class_embed.weight"
    if failure == "missing":
        state.pop(dn_key)
        expected_error = "Missing checkpoint key"
    else:
        state[dn_key] = torch.randn(1)
        expected_error = "shape/dtype mismatch"
    before_detector = wrapper.model.state_dict()[detector_key].detach().clone()
    before_embedding = wrapper.denoising.class_embed.weight.detach().clone()
    path = tmp_path / f"invalid-dn-{failure}.pth"
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match=expected_error):
        load_official_model_weights(
            path,
            model=wrapper,
            source="model",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    torch.testing.assert_close(wrapper.model.state_dict()[detector_key], before_detector)
    torch.testing.assert_close(wrapper.denoising.class_embed.weight, before_embedding)
