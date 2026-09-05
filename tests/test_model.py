# Copyright 2026 Clean RT-DETR contributors
"""Integration contracts for image-to-prediction composition."""

import pytest
import torch

from clean_rtdetr import (
    RTDETR,
    Decoder,
    DecoderConfig,
    DecoderPlan,
    FeatureSpec,
    HybridEncoder,
    HybridEncoderConfig,
    MemoryProjector,
    ModelOutput,
    PResNet,
    QuerySelector,
    QuerySelectorConfig,
    build_rtdetr_r50vd,
)


def _small_model() -> RTDETR:
    """Keep the real backbone and use small downstream dimensions for autograd checks."""
    backbone = PResNet()
    encoder = HybridEncoder(
        backbone.specs,
        HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
    )
    return RTDETR(
        backbone,
        encoder,
        MemoryProjector(encoder.output_specs, hidden_dim=8),
        QuerySelector(QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=4)),
        Decoder(
            DecoderConfig(
                hidden_dim=8, num_classes=3, num_layers=2, num_heads=2, feedforward_dim=16
            )
        ),
    )


def test_composition_taps_and_execution_plan() -> None:
    """Feature collection preserves predictions and the decoder plan passes through."""
    model = _small_model().eval()
    images = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        regular = model(images)
        tapped = model(images, collect_aifi=True)
        prefix = model(images, plan=DecoderPlan(num_layers=1, collect_intermediate=False))
    assert regular.after_aifi is None
    assert tapped.after_aifi is not None
    assert tapped.after_aifi.specs == (FeatureSpec(8, 32),)
    assert tapped.after_aifi.levels[0].shape == (1, 8, 2, 3)
    torch.testing.assert_close(regular.decoder.logits, tapped.decoder.logits, atol=0, rtol=0)
    torch.testing.assert_close(regular.decoder.boxes, tapped.decoder.boxes, atol=0, rtol=0)
    torch.testing.assert_close(prefix.final.logits, regular.decoder.logits[0])
    assert prefix.decoder.logits.shape == (1, 1, 4, 3)
    assert prefix.final.boxes.shape == (1, 4, 4)
    assert bool(((prefix.final.boxes >= 0) & (prefix.final.boxes <= 1)).all())
    assert len(list(model.parameters())) == len({id(value) for value in model.parameters()})


def test_training_output_and_auxiliary_gradients() -> None:
    """Keep stable fields and gradients through AIFI and encoder auxiliary predictions."""
    model = _small_model().train()
    images = torch.rand(2, 3, 64, 96, requires_grad=True)
    output = model(images, collect_aifi=True)
    assert isinstance(output, ModelOutput)
    assert output.after_aifi is not None
    feature = output.after_aifi.levels[0]
    feature.retain_grad()
    loss = (
        output.final.logits.square().mean()
        + output.encoder_aux.logits.square().mean()
        + output.encoder_aux.boxes.mean()
        + feature.square().mean()
    )
    loss.backward()
    assert feature.grad is not None
    assert bool(feature.grad.abs().sum() > 0)
    assert images.grad is not None
    assert bool(images.grad.abs().sum() > 0)
    assert all(parameter.grad is None for parameter in model.backbone.conv1.parameters())
    assert any(parameter.grad is not None for parameter in model.backbone.res_layers.parameters())
    assert any(
        parameter.grad is not None for parameter in model.selector.enc_bbox_head.parameters()
    )
    assert all(parameter.grad is not None for parameter in model.encoder.encoder.parameters())


def test_factory_defaults_and_insufficient_queries() -> None:
    """Build the official dimensions and reject inputs with fewer than 300 feature points."""
    model = build_rtdetr_r50vd().eval()
    assert model.training is False
    assert model.selector.config == QuerySelectorConfig()
    assert model.decoder.config == DecoderConfig()
    assert model.encoder.output_specs == tuple(FeatureSpec(256, stride) for stride in (8, 16, 32))
    with torch.no_grad(), pytest.raises(ValueError, match="at least num_queries"):
        model(torch.rand(1, 3, 64, 64))


@pytest.mark.parametrize("boundary", ["features", "hidden", "classes", "levels"])
def test_composition_rejects_incompatible_modules(boundary: str) -> None:
    """Reject component contracts that cannot be connected without reinterpretation."""
    model = _small_model()
    projector = model.projector
    selector = model.selector
    decoder = model.decoder
    if boundary == "features":
        projector = MemoryProjector(tuple(FeatureSpec(9, stride) for stride in (8, 16, 32)), 8)
    elif boundary == "hidden":
        projector = MemoryProjector(model.encoder.output_specs, 16)
    elif boundary == "classes":
        selector = QuerySelector(QuerySelectorConfig(hidden_dim=8, num_classes=4, num_queries=4))
    else:
        decoder = Decoder(DecoderConfig(hidden_dim=8, num_classes=3, num_levels=2, num_heads=2))
    with pytest.raises(ValueError, match=r"specifications|incompatible"):
        RTDETR(model.backbone, model.encoder, projector, selector, decoder)
