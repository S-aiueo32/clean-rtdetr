# Copyright 2026 Clean RT-DETR contributors
"""Exercise train-only distillation, optimizer ownership and inference isolation."""

from functools import partial

import pytest
import torch

from clean_rtdetr import (
    RTDETR,
    Decoder,
    DecoderConfig,
    HybridEncoder,
    HybridEncoderConfig,
    MemoryProjector,
    PResNet,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr_training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    DetectionTarget,
    HungarianMatcher,
    SetCriterion,
)
from clean_rtdetr_training.lightning import RTDETRLightningModule
from rtdetrv4_recipe.criterion import DistillationCriterion
from rtdetrv4_recipe.model import DistillationTrainingModel
from rtdetrv4_recipe.teacher import SmokeTeacher


def make_model() -> DistillationTrainingModel:
    """Shrink attention and queries while preserving the public training boundaries."""
    backbone = PResNet(depth=18)
    encoder = HybridEncoder(
        backbone.specs, HybridEncoderConfig(hidden_dim=16, num_heads=4, feedforward_dim=32)
    )
    model = RTDETR(
        backbone,
        encoder,
        MemoryProjector(encoder.output_specs, hidden_dim=16),
        QuerySelector(QuerySelectorConfig(hidden_dim=16, num_classes=1, num_queries=4)),
        QueryAssembler(),
        Decoder(
            DecoderConfig(
                hidden_dim=16,
                num_classes=1,
                num_layers=2,
                num_heads=4,
                num_points=1,
                feedforward_dim=32,
            )
        ),
    )
    student = DenoisingTrainingModel.from_factory(
        model, partial(DenoisingQueryGenerator, config=DenoisingConfig(num_denoising=4))
    )
    return DistillationTrainingModel(student, SmokeTeacher(channels=8))


@pytest.mark.parametrize("empty", [False, True])
def test_training_gradients_and_optimizer_ownership(*, empty: bool) -> None:
    """DSI reaches AIFI/backbone/projector; DN is retained and the teacher stays frozen."""
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = make_model().train()
    images = torch.rand(2, 3, 64, 64)
    target = (
        DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
        if empty
        else DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.3]]))
    )
    criterion = DistillationCriterion(SetCriterion(matcher=HungarianMatcher()))
    output = model(images, targets=[target, target])
    losses = criterion(output, [target, target])
    assert "loss_distill" in losses
    assert "loss_vfl_aux_0" in losses
    assert ("loss_vfl_dn_0" in losses) is not empty
    losses["loss_distill"].backward(retain_graph=True)
    for part in (
        model.student.model.backbone,
        model.student.model.encoder.encoder,
        model.projector,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    sum(losses.values()).backward()
    assert not model.teacher.training
    assert all(not p.requires_grad and p.grad is None for p in model.teacher.parameters())
    module = RTDETRLightningModule(model, criterion, partial(torch.optim.AdamW, lr=1e-4))
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(p) for p in module.parameters() if p.requires_grad}
    optimizer.step()


def test_predict_skips_teacher_and_projector_and_restores_modes() -> None:
    """Student predictions exactly match the detector without running training extras."""
    model = make_model().train()
    images = torch.rand(1, 3, 64, 64)

    def reject_call(*_args: object) -> None:
        """Fail if an inference request evaluates either training-only module."""
        pytest.fail("Inference ran a training-only component")

    handles = [
        part.register_forward_pre_hook(reject_call) for part in (model.teacher, model.projector)
    ]
    before = [part.training for part in model.modules()]
    with torch.no_grad():
        prediction = model.predict(images)
        assert before == [part.training for part in model.modules()]
        model.eval()
        expected = model.student.model(images)
        torch.testing.assert_close(prediction.final.logits, expected.final.logits)
        torch.testing.assert_close(prediction.final.boxes, expected.final.boxes)
        assert model(images).student is None
    for handle in handles:
        handle.remove()
    criterion = DistillationCriterion(SetCriterion(matcher=HungarianMatcher()))
    with pytest.raises(ValueError, match="training-time"):
        criterion(prediction, [])
