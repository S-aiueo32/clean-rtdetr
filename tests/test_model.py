# Copyright 2026 Clean RT-DETR contributors
"""Integration checks for the explicit image-to-prediction composition."""

from dataclasses import replace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from clean_rtdetr import (
    RTDETR,
    Decoder,
    DecoderConfig,
    DecoderPlan,
    FeatureSpec,
    HybridEncoder,
    HybridEncoderConfig,
    MemoryProjector,
    PResNet,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr.configuration import compose_model_config
from clean_rtdetr.models.rtdetr import BaselineRTDETR

BASELINE_CLASSES = 80
BASELINE_QUERIES = 300
BASELINE_LAYERS = 6
BASELINE_HIDDEN = 256
BASELINE_LEVELS = 3
OVERRIDE_QUERIES = 20


def small_model() -> BaselineRTDETR:
    """Build a compact model retaining all production boundaries."""
    backbone = PResNet()
    encoder = HybridEncoder(
        backbone.specs,
        HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
    )
    projector = MemoryProjector(encoder.output_specs, hidden_dim=8)
    selector = QuerySelector(QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=2))
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=8,
            num_classes=3,
            num_layers=2,
            num_heads=2,
            num_levels=3,
            num_points=1,
            feedforward_dim=16,
        )
    )
    return RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)


def baseline_model() -> BaselineRTDETR:
    """Instantiate the packaged pinned R50vd model recipe."""
    return instantiate(compose_model_config("r50vd"), _convert_="all")


def test_model_matches_manual_stage_composition() -> None:
    """The wrapper must preserve the outputs of its injected stages exactly."""
    torch.manual_seed(5)
    model = small_model().eval()
    images = torch.randn(1, 3, 64, 64)
    actual = model(images, plan=DecoderPlan(num_layers=1))
    features = model.backbone(images)
    encoded_features = model.encoder(features)
    encoded = model.projector(encoded_features.features)
    selected = model.selector(encoded)
    queries = model.assembler(selected.queries)
    expected = model.decoder(queries, encoded, plan=DecoderPlan(num_layers=1))
    torch.testing.assert_close(actual.decoder.logits, expected.logits)
    torch.testing.assert_close(actual.decoder.boxes, expected.boxes)
    torch.testing.assert_close(actual.encoder_aux.logits, selected.encoder_logits)


def test_model_validates_component_contracts() -> None:
    """Reject mismatched metadata and dimensions during construction."""
    model = small_model()
    with pytest.raises(ValueError, match="feature specs"):
        RTDETR(
            model.backbone,
            HybridEncoder(
                (FeatureSpec(512, 8), FeatureSpec(1024, 16), FeatureSpec(1024, 32)),
                HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
            ),
            model.projector,
            model.selector,
            model.assembler,
            model.decoder,
        )


def test_model_forwards_plan_and_aifi_without_changing_predictions() -> None:
    """AIFI capture is observational and decoder plans select layer prefixes."""
    torch.manual_seed(9)
    model = small_model().eval()
    images = torch.randn(1, 3, 64, 64)
    plain = model(images, plan=DecoderPlan(num_layers=2), collect_aifi=False)
    tapped = model(images, plan=DecoderPlan(num_layers=2), collect_aifi=True)
    assert tapped.after_aifi is not None
    torch.testing.assert_close(plain.decoder.logits, tapped.decoder.logits)
    assert model(images, plan=DecoderPlan(num_layers=1)).decoder.logits.shape[0] == 1
    with pytest.raises(ValueError, match="num_layers"):
        model(images, plan=DecoderPlan(num_layers=0))


def test_model_rejects_invalid_image_before_downstream_work() -> None:
    """Preserve the backbone input contract at the public model boundary."""
    model = small_model()
    with pytest.raises(ValueError, match="multiples of 32"):
        model(torch.randn(1, 3, 63, 64))


@pytest.mark.parametrize(
    ("kind", "match"),
    [("hidden", "hidden dimensions"), ("classes", "class counts"), ("levels", "level count")],
)
def test_model_rejects_each_decoder_contract_mismatch(kind: str, match: str) -> None:
    """Validate decoder hidden size, class count, and feature level count independently."""
    model = small_model()
    changes = {
        "hidden": {"hidden_dim": 16},
        "classes": {"num_classes": 4},
        "levels": {"num_levels": 2},
    }
    replacement = Decoder(replace(model.decoder.config, **changes[kind]))
    with pytest.raises(ValueError, match=match):
        RTDETR(
            model.backbone,
            model.encoder,
            model.projector,
            model.selector,
            model.assembler,
            replacement,
        )


@pytest.mark.parametrize("loss_source", ["encoder_aux", "aifi"])
def test_auxiliary_and_aifi_losses_retain_gradient_routes(loss_source: str) -> None:
    """Auxiliary predictions and captured AIFI features remain connected to the model."""
    torch.manual_seed(12)
    model = small_model().train()
    images = torch.randn(1, 3, 64, 64, requires_grad=True)
    output = model(images, collect_aifi=True)
    assert output.after_aifi is not None
    loss = (
        output.encoder_aux.logits.square().mean() + output.encoder_aux.boxes.square().mean()
        if loss_source == "encoder_aux"
        else output.after_aifi.levels[0].square().mean()
    )
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    for module in (model.encoder, model.backbone):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_decoder_loss_does_not_backpropagate_through_assembled_references() -> None:
    """The query assembly boundary detaches selector reference logits from decoding."""
    model = small_model().train()
    images = torch.randn(1, 3, 64, 64)
    features = model.backbone(images)
    encoded_features = model.encoder(features)
    encoded = model.projector(encoded_features.features)
    selected = model.selector(encoded)
    selected.queries.ref_logits.retain_grad()
    queries = model.assembler(selected.queries)
    model.decoder(queries, encoded).logits.sum().backward()
    assert selected.queries.ref_logits.grad is None


def test_r50vd_config_uses_pinned_baseline_dimensions() -> None:
    """The packaged recipe exposes the planned baseline dimensions."""
    config = compose_model_config("r50vd")
    assert "examples." not in OmegaConf.to_yaml(config, resolve=True)
    model = instantiate(config, _convert_="all")
    assert model.selector.config.num_classes == BASELINE_CLASSES
    assert model.selector.config.num_queries == BASELINE_QUERIES
    assert model.selector.config.learned_content is False
    assert model.decoder.config.num_layers == BASELINE_LAYERS
    assert model.decoder.config.hidden_dim == BASELINE_HIDDEN
    assert model.decoder.config.num_levels == BASELINE_LEVELS


def test_r50vd_config_accepts_hydra_model_overrides() -> None:
    """Packaged model settings remain selectable without a Python builder."""
    config = compose_model_config(
        "r50vd", [f"model.selector.config.num_queries={OVERRIDE_QUERIES}"]
    )
    model = instantiate(config, _convert_="all")

    assert model.selector.config.num_queries == OVERRIDE_QUERIES
