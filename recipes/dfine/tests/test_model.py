# Copyright 2026 Clean RT-DETR contributors
"""Check FDR integration, gradient routes, prediction mode and loss contracts."""

from dataclasses import replace

import pytest
import torch

from clean_rtdetr import (
    RTDETR,
    DecoderConfig,
    HybridEncoder,
    HybridEncoderConfig,
    MemoryProjector,
    PResNet,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr.contracts import DecoderPlan
from clean_rtdetr_training import DetectionTarget
from dfine_recipe.criterion import DFINECriterion, DFINECriterionConfig
from dfine_recipe.decoder import DistributionOutput, FDRDecoder
from dfine_recipe.matcher import DFINEMatcher
from dfine_recipe.model import DFINETrainingModel


def _model(*, use_lqe: bool) -> DFINETrainingModel:
    """Keep production boundaries while shrinking hidden width and query count."""
    backbone = PResNet(depth=18)
    encoder = HybridEncoder(
        backbone.specs, HybridEncoderConfig(hidden_dim=16, num_heads=4, feedforward_dim=32)
    )
    projector = MemoryProjector(encoder.output_specs, hidden_dim=16)
    selector = QuerySelector(QuerySelectorConfig(hidden_dim=16, num_classes=2, num_queries=4))
    decoder = FDRDecoder(
        DecoderConfig(
            hidden_dim=16,
            num_classes=2,
            num_layers=2,
            num_heads=4,
            num_points=1,
            feedforward_dim=32,
        ),
        reg_max=8,
        use_lqe=use_lqe,
    )
    return DFINETrainingModel(
        RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)
    )


@pytest.mark.parametrize("use_lqe", [False, True])
def test_forward_backward_and_plan(*, use_lqe: bool) -> None:
    """Pre-box and distribution heads train; fixed references remain detached."""
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = _model(use_lqe=use_lqe)
    images = torch.rand(2, 3, 64, 64)
    targets = [
        DetectionTarget(torch.tensor([1]), torch.tensor([[0.5, 0.5, 0.2, 0.3]])),
        DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4)),
    ]
    output = model(images, targets=targets)
    decoded = output.decoder
    assert isinstance(decoded, DistributionOutput)
    assert decoded.corners.shape == (2, 2, 4, 4, 9)
    assert not decoded.references.requires_grad
    assert decoded.pre.boxes.requires_grad
    criterion = DFINECriterion(DFINEMatcher(), DFINECriterionConfig(reg_max=8))
    losses = criterion(output, targets)
    assert {
        "loss_fgl",
        "loss_ddf",
        "loss_ddf_aux_0",
        "loss_vfl_pre",
        "loss_vfl_enc",
    } <= losses.keys()
    sum(losses.values()).backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    for part in ("pre_bbox_head", "distribution_heads.0", "distribution_heads.1"):
        assert any(
            grad is not None and grad.abs().sum() > 0
            for name, grad in grads.items()
            if part in name
        )
    with torch.no_grad():
        prediction = model.predict(images)
        assert model.training
        assert isinstance(prediction.decoder, DistributionOutput)
        model.eval()
        prefix = model(images, plan=DecoderPlan(num_layers=1, collect_intermediate=False))
        assert prefix.decoder.logits.shape[0] == 1
        assert isinstance(prefix.decoder, DistributionOutput)
        prefix.decoder.validate()


def test_distribution_settings_and_invalid_plans_are_rejected() -> None:
    """Changing the bin geometry cannot silently change the training objective."""
    model = _model(use_lqe=False).eval()
    images = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        output = model(images)
    changed = DFINECriterion(DFINEMatcher(), DFINECriterionConfig(reg_max=32))
    targets = [DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))]
    with pytest.raises(ValueError, match="settings must match"):
        changed(output, targets)
    with pytest.raises(ValueError, match="configured prefix"):
        model(images, plan=DecoderPlan(num_layers=0))
    config = DFINECriterionConfig(reg_max=8)
    assert (
        DFINECriterion(DFINEMatcher(), config).checkpoint_config()
        != DFINECriterion(DFINEMatcher(), replace(config, temperature=3.0)).checkpoint_config()
    )
