# Copyright 2026 Clean RT-DETR contributors
"""Lightning training against injected model and criterion interfaces."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import torch
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn

from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.interfaces import TrainingCriterion, TrainingModel

TrainingBatch = tuple[Float[Tensor, "batch 3 height width"], list[DetectionTarget]]
OptimizerFactory = Callable[[Iterable[Any]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
ParameterGroupsFactory = Callable[[nn.Module], list[dict[str, Any]]]
PredictionT = TypeVar("PredictionT")


def _factory_recipe(factory: object) -> dict[str, Any] | None:
    """Describe Hydra partials without pickling runtime factory objects."""
    if factory is None:
        return None
    if isinstance(factory, partial):
        return {
            "target": _factory_recipe(factory.func),
            "args": factory.args,
            "kwargs": factory.keywords,
        }
    if is_dataclass(factory) and not isinstance(factory, type):
        return {"target": _factory_recipe(type(factory)), "config": asdict(factory)}
    if callable(factory) and hasattr(factory, "__qualname__"):
        return {"target": f"{factory.__module__}.{factory.__qualname__}"}
    msg = "Checkpointable factories must be named callables, dataclasses, or partials"
    raise TypeError(msg)


class RTDETRLightningModule(LightningModule, Generic[PredictionT]):
    """Keep the detector reusable while delegating loops and resume to Lightning."""

    def __init__(
        self,
        training_model: TrainingModel[PredictionT],
        criterion: TrainingCriterion[PredictionT],
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None = None,
        parameter_groups_factory: ParameterGroupsFactory | None = None,
    ) -> None:
        """Register interface implementations and factories as ordinary module dependencies."""
        super().__init__()
        self.training_model = training_model
        self.criterion = criterion
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.parameter_groups_factory = parameter_groups_factory
        for factory in (optimizer_factory, scheduler_factory, parameter_groups_factory):
            _factory_recipe(factory)
        self._resume_execution: dict[str, Any] | None = None
        self._resume_callback_keys: set[str] = set()

    def forward(self, images: Float[Tensor, "batch 3 height width"]) -> PredictionT:
        """Delegate target-free prediction without assuming an internal model layout."""
        return self.training_model.predict(images)

    def transfer_batch_to_device(
        self, batch: TrainingBatch, device: torch.device, dataloader_idx: int
    ) -> TrainingBatch:
        """Rebuild immutable targets rather than asking Lightning to mutate dataclasses."""
        del dataloader_idx
        images, targets = batch
        return images.to(device), [
            DetectionTarget(target.labels.to(device), target.boxes.to(device)) for target in targets
        ]

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Float[Tensor, ""]:
        """Sum the injected criterion's weighted losses for automatic optimization."""
        del batch_idx
        images, targets = batch
        output = self.training_model(images, targets=targets)
        losses = self.criterion(output, targets)
        total = torch.stack(list(losses.values())).sum()
        if not bool(torch.isfinite(total)):
            msg = "Training loss must be finite"
            raise ValueError(msg)
        self.log("train/loss", total, on_step=True, on_epoch=True, batch_size=images.shape[0])
        return total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Bind runtime parameters to the injected optimizer and scheduler factories."""
        parameters = (
            self.parameter_groups_factory(self.training_model)
            if self.parameter_groups_factory is not None
            else [{"params": [p for p in self.training_model.parameters() if p.requires_grad]}]
        )
        model_parameters = {id(p) for p in self.training_model.parameters()}
        criterion_parameters = [
            p
            for p in self.criterion.parameters()
            if p.requires_grad and id(p) not in model_parameters
        ]
        if criterion_parameters:
            parameters = [*parameters, {"params": criterion_parameters}]
        optimizer = self.optimizer_factory(parameters)
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if set(actual) != expected or len(actual) != len(expected):
            msg = "Optimizer must own every trainable parameter exactly once"
            raise ValueError(msg)
        if self.scheduler_factory is None:
            return optimizer
        scheduler = self.scheduler_factory(optimizer)
        if scheduler.optimizer is not optimizer:
            msg = "Scheduler must reference the injected optimizer"
            raise ValueError(msg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_fit_start(self) -> None:
        """Fail early for execution modes outside the current loss and resume scope."""
        trainer = self.trainer
        if trainer.world_size != 1 or self.device.type != "cpu":
            msg = "Lightning training currently supports single-process CPU execution only"
            raise ValueError(msg)
        if trainer.precision != "32-true":
            msg = "Lightning training currently requires 32-true precision"
            raise ValueError(msg)
        if trainer.accumulate_grad_batches != 1:
            msg = "Gradient accumulation is not yet supported by the baseline recipe"
            raise ValueError(msg)
        if self._resume_execution is not None and self._resume_execution != self._execution():
            msg = "Checkpoint Trainer execution settings do not match"
            raise ValueError(msg)
        if not self._resume_callback_keys <= {
            callback.state_key for callback in getattr(trainer, "callbacks", ())
        }:
            msg = "Resume requires the checkpoint's EMA and epoch RNG callbacks"
            raise ValueError(msg)

    def _execution(self) -> dict[str, Any]:
        """Retain Trainer settings that change optimizer updates across a restart."""
        return {
            "precision": self.trainer.precision,
            "gradient_clip_val": self.trainer.gradient_clip_val,
            "gradient_clip_algorithm": str(self.trainer.gradient_clip_algorithm),
            "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
        }

    def _recipe(self) -> dict[str, Any]:
        """Record settings that tensor shapes alone cannot validate during resume."""
        model_type = type(self.training_model)
        criterion_type = type(self.criterion)
        return {
            "version": 2,
            "optimizer": _factory_recipe(self.optimizer_factory),
            "scheduler": _factory_recipe(self.scheduler_factory),
            "parameter_groups": _factory_recipe(self.parameter_groups_factory),
            "training_model": {
                "target": f"{model_type.__module__}.{model_type.__qualname__}",
                "config": self.training_model.checkpoint_config(),
            },
            "criterion": {
                "target": f"{criterion_type.__module__}.{criterion_type.__qualname__}",
                "config": self.criterion.checkpoint_config(),
            },
            "trainable_parameters": tuple(
                name for name, parameter in self.named_parameters() if parameter.requires_grad
            ),
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Let Lightning own serialization while adding the reproduction settings."""
        checkpoint["clean_rtdetr_recipe"] = self._recipe()
        checkpoint["clean_rtdetr_execution"] = self._execution()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject changed training settings instead of silently resuming another recipe."""
        if checkpoint.get("clean_rtdetr_recipe") != self._recipe():
            msg = "Checkpoint training recipe does not match this Lightning module"
            raise ValueError(msg)
        self._resume_execution = checkpoint["clean_rtdetr_execution"]
        self._resume_callback_keys = {
            key
            for key in checkpoint.get("callbacks", {})
            if key in {"RTDETREMA", "EpochBoundaryState"}
        }
