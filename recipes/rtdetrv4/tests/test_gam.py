# Copyright 2026 Clean RT-DETR contributors
"""Check the published GAM update rule and gradient component accounting."""

from collections.abc import Sequence
from functools import partial

import pytest
import torch
from lightning.pytorch import Trainer
from test_model import make_model

from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion
from clean_rtdetr_training.contracts import TrainingOutput
from clean_rtdetr_training.interfaces import TrainingCriterion, TrainingLosses
from clean_rtdetr_training.lightning import RTDETRLightningModule
from rtdetrv4_recipe.criterion import DistillationCriterion
from rtdetrv4_recipe.gam import GradientAdaptiveModulation
from rtdetrv4_recipe.model import DistillationTrainingModel


class TrainingOnlyCriterion(TrainingCriterion[TrainingOutput]):
    """Minimal detection criterion proving v4 composition needs TrainingOutput only."""

    def forward(self, output: TrainingOutput, targets: Sequence[DetectionTarget]) -> TrainingLosses:
        """Return one detection loss from the canonical training payload."""
        del targets
        return {"loss_detection": output.final.logits.square().mean()}

    def checkpoint_config(self) -> dict[str, object]:
        """Describe this test criterion's fixed configuration."""
        return {"criterion": "training-only"}


def _module() -> RTDETRLightningModule:
    """Construct the actual injected graph used by the callback."""
    return RTDETRLightningModule(
        make_model(),
        DistillationCriterion(SetCriterion(matcher=HungarianMatcher())),
        partial(torch.optim.AdamW, lr=1e-4),
    )


def test_training_output_only_detection_criterion_composes() -> None:
    """Distillation accepts a detector criterion whose input is TrainingOutput."""
    torch.set_num_threads(1)
    model = make_model().train()
    target = DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.3]]))
    output = model(torch.rand(1, 3, 64, 64), targets=[target])
    losses = DistillationCriterion(TrainingOnlyCriterion())(output, [target])
    assert set(losses) == {"loss_detection", "loss_distill"}


@pytest.mark.parametrize(
    ("percentage", "expected"),
    [(1.0, 31.8539), (3.0, 9.3935), (2.0, 15.0), (0.0, 7.0), (0.0001, 150.0), (99.0, 1.5)],
)
def test_official_weight_update(percentage: float, expected: float) -> None:
    """Check both opposite boundaries, in-band, reset and factor clamps."""
    callback = GradientAdaptiveModulation(default_weight=7.0)
    callback.load_state_dict({**callback.state_dict(), "count": 1, "percentage_sum": percentage})
    module = _module()
    callback.on_train_epoch_end(Trainer(logger=False), module)
    assert isinstance(module.criterion, DistillationCriterion)
    assert float(module.criterion.distillation_weight) == pytest.approx(expected, rel=1e-4)


def test_stop_epoch_resets_weight() -> None:
    """The caller's augmentation stop epoch restores the configured fixed weight."""
    callback = GradientAdaptiveModulation(default_weight=7.0, stop_epoch=0)
    callback.load_state_dict({**callback.state_dict(), "count": 1, "percentage_sum": 1.0})
    module = _module()
    callback.on_train_epoch_end(Trainer(logger=False), module)
    assert isinstance(module.criterion, DistillationCriterion)
    assert float(module.criterion.distillation_weight) == pytest.approx(7.0)


def test_gradient_percentage_uses_aifi_and_all_trainable_parameters() -> None:
    """The projector contributes to the denominator and frozen teacher does not."""
    callback = GradientAdaptiveModulation()
    module = _module()
    trainer = Trainer(logger=False)
    callback.on_fit_start(trainer, module)
    assert isinstance(module.training_model, DistillationTrainingModel)
    aifi = next(module.training_model.student.model.encoder.aifi_parameters())
    projector = module.training_model.projector.weight
    aifi.grad = torch.zeros_like(aifi)
    aifi.grad.flatten()[0] = 1.0
    projector.grad = torch.zeros_like(projector)
    projector.grad.flatten()[0] = 3.0
    callback.on_before_optimizer_step(trainer, module, torch.optim.SGD(module.parameters()))
    state = callback.state_dict()
    assert state["count"] == 1
    assert state["percentage_sum"] == pytest.approx(25.0)
    callback.on_train_epoch_start(trainer, module)
    assert callback.state_dict()["count"] == 0


def test_state_restore_rejects_changed_settings() -> None:
    """Validate state and constructor settings on restart."""
    callback = GradientAdaptiveModulation(rho=3.0)
    state = {**callback.state_dict(), "count": 4, "percentage_sum": 12.5}
    callback.load_state_dict(state)
    assert callback.state_dict() == state
    with pytest.raises(ValueError, match="rho"):
        GradientAdaptiveModulation().load_state_dict(state)
    with pytest.raises(ValueError, match="percentage_sum"):
        callback.load_state_dict({**state, "percentage_sum": float("nan")})


@pytest.mark.parametrize(
    ("rho", "delta", "weight"),
    [(0.0, 0.1, 15.0), (2.0, -1.0, 15.0), (float("nan"), 0.1, 15.0), (2.0, 0.1, 0.0)],
)
def test_invalid_settings_are_rejected(rho: float, delta: float, weight: float) -> None:
    """Fail before creating an optimizer when modulation settings are invalid."""
    with pytest.raises(ValueError, match="GAM"):
        GradientAdaptiveModulation(rho=rho, delta=delta, default_weight=weight)
