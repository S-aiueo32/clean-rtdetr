# Copyright 2026 Clean RT-DETR contributors
"""Gradient Adaptive Modulation for RT-DETRv4 distillation."""

from __future__ import annotations

import math
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor

from clean_rtdetr.models import HybridEncoder

from .criterion import DistillationCriterion
from .model import DistillationTrainingModel

PERCENT = 100.0
DENOMINATOR_EPSILON = 1e-9


class GradientAdaptiveModulation(Callback):
    """Adapt the distillation weight to the AIFI gradient share.

    The update follows RT-DETRv4's odds-ratio rule: an observed percentage
    outside ``rho ± delta`` is moved toward the opposite bound, with the resulting
    weight factor limited to one tenth through ten times its current value.
    ``rho`` and ``delta`` are percentages. Updates use the epoch mean; a zero
    mean restores ``default_weight``. If set, ``stop_epoch`` is zero-based and
    restores that default at the end of that epoch and every later epoch.
    """

    def __init__(
        self,
        rho: float = 2.0,
        delta: float = 0.1,
        default_weight: float = 15.0,
        stop_epoch: int | None = None,
    ) -> None:
        """Configure the target gradient share and adaptive weight settings."""
        super().__init__()
        self.rho = rho
        self.delta = delta
        self.default_weight = default_weight
        self.stop_epoch = stop_epoch
        self._percentage_sum = 0.0
        self._count = 0
        self._validate_settings()

    @property
    def state_key(self) -> str:
        """Use a stable key so Lightning can resume callback state."""
        return "GradientAdaptiveModulation"

    def _validate_settings(self) -> None:
        values = (self.rho, self.delta, self.default_weight)
        if not all(math.isfinite(value) for value in values):
            msg = "GAM settings must be finite"
            raise ValueError(msg)
        if not (0 < self.rho - self.delta <= self.rho + self.delta < PERCENT):
            msg = "GAM requires 0 < rho-delta <= rho+delta < 100"
            raise ValueError(msg)
        if self.default_weight <= 0:
            msg = "GAM default_weight must be positive"
            raise ValueError(msg)
        if self.stop_epoch is not None and (
            type(self.stop_epoch) is not int or self.stop_epoch < 0
        ):
            msg = "GAM stop_epoch must be a non-negative integer or None"
            raise ValueError(msg)

    @staticmethod
    def _components(pl_module: LightningModule) -> tuple[HybridEncoder, DistillationCriterion]:
        """Narrow the injected graph at the callback boundary."""
        if not isinstance(pl_module.training_model, DistillationTrainingModel):
            msg = "GAM requires DistillationTrainingModel"
            raise TypeError(msg)
        if not isinstance(pl_module.criterion, DistillationCriterion):
            msg = "GAM requires DistillationCriterion"
            raise TypeError(msg)
        if not isinstance(pl_module.training_model.student.model.encoder, HybridEncoder):
            msg = "GAM requires a HybridEncoder"
            raise TypeError(msg)
        weight = pl_module.criterion.distillation_weight
        if not isinstance(weight, Tensor) or weight.ndim != 0 or not bool(torch.isfinite(weight)):
            msg = "Distillation weight must be a finite scalar tensor"
            raise TypeError(msg)
        if float(weight) <= 0:
            msg = "Distillation weight must be positive"
            raise ValueError(msg)
        return pl_module.training_model.student.model.encoder, pl_module.criterion

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Validate the callback and recipe object graph before fitting."""
        del trainer
        self._validate_settings()
        self._components(pl_module)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Accumulate the AIFI gradient share for the current epoch."""
        del trainer, optimizer
        encoder, _ = self._components(pl_module)
        aifi_parameters = {id(parameter) for parameter in encoder.aifi_parameters()}
        total = 0.0
        aifi = 0.0
        for parameter in pl_module.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            magnitude = float(parameter.grad.detach().abs().sum())
            total += magnitude
            if id(parameter) in aifi_parameters:
                aifi += magnitude
        percentage = PERCENT * aifi / total if total > 0 else 0.0
        self._percentage_sum += percentage
        self._count += 1

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start a fresh gradient share average for each training epoch."""
        del trainer, pl_module
        self._percentage_sum = 0.0
        self._count = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply the official odds ratio update to the distillation weight."""
        del trainer
        _, criterion = self._components(pl_module)
        epoch = pl_module.current_epoch
        average = self._percentage_sum / self._count if self._count else 0.0
        weight = criterion.distillation_weight
        if average == 0.0 or (self.stop_epoch is not None and epoch >= self.stop_epoch):
            with torch.no_grad():
                weight.fill_(self.default_weight)
            return
        lower = self.rho - self.delta
        upper = self.rho + self.delta
        if lower <= average <= upper:
            return
        target = upper if average < lower else lower
        current = float(weight)
        p_current = average / PERCENT
        p_target = target / PERCENT
        denominator = p_current * (1.0 - p_target)
        if abs(denominator) < DENOMINATOR_EPSILON:
            return
        factor = (p_target * (1.0 - p_current)) / denominator
        factor = min(max(factor, 0.1), 10.0)
        with torch.no_grad():
            weight.fill_(current * factor)

    def state_dict(self) -> dict[str, Any]:
        """Serialize running observations and immutable callback settings."""
        return {
            "rho": self.rho,
            "delta": self.delta,
            "default_weight": self.default_weight,
            "stop_epoch": self.stop_epoch,
            "count": self._count,
            "percentage_sum": self._percentage_sum,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state while rejecting a callback configured differently."""
        self._validate_settings()
        for name in ("rho", "delta", "default_weight", "stop_epoch"):
            if state_dict.get(name) != getattr(self, name):
                msg = f"GAM setting {name} does not match checkpoint"
                raise ValueError(msg)
        count = state_dict.get("count")
        percentage_sum = state_dict.get("percentage_sum")
        if type(count) is not int or count < 0:
            msg = "GAM checkpoint count must be a non-negative integer"
            raise ValueError(msg)
        if (
            not isinstance(percentage_sum, (int, float))
            or not math.isfinite(percentage_sum)
            or not 0 <= percentage_sum <= PERCENT * count
        ):
            msg = "GAM checkpoint percentage_sum must be finite"
            raise ValueError(msg)
        self._count = count
        self._percentage_sum = float(percentage_sum)
