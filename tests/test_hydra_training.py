# Copyright 2026 Clean RT-DETR contributors
"""Regression checks for the public Hydra training composition."""

from pathlib import Path
from typing import Any

import lightning
import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from clean_rtdetr import RTDETR
from clean_rtdetr.configuration import compose_config
from clean_rtdetr.training.lightning import RTDETRLightningModule
from tests.test_training_interfaces import (
    FixedMatcher,
    ToyAssembler,
    ToyCriterion,
    ToyTrainingModel,
    _loader,
)


def _compose(overrides: list[str] | None = None) -> DictConfig:
    return compose_config("lightning_training", overrides or [])


def _targets(value: Any):  # noqa: ANN202, ANN401 - recursive test helper
    if hasattr(value, "items"):
        for key, child in value.items():
            if key == "_target_":
                yield str(child)
            yield from _targets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _targets(child)


def test_lightning_config_composes_full_model_without_command_targets() -> None:
    """The composed object graph builds the package model directly."""
    config = _compose()
    targets = tuple(_targets(OmegaConf.to_container(config, resolve=True)))
    assert targets
    assert all(not target.startswith("clean_rtdetr.cli.") for target in targets)
    module = instantiate(config.module, _convert_="all")
    assert isinstance(module, RTDETRLightningModule)
    assert isinstance(module.training_model.model, RTDETR)
    assert module.training_model.model.encoder.specs == module.training_model.model.backbone.specs


def test_lightning_config_group_overrides_build_optimizer_and_scheduler() -> None:
    """Config groups replace factories and bind them to the actual model parameters."""
    config = _compose(
        ["optimizer@module.optimizer_factory=sgd", "scheduler@module.scheduler_factory=step"]
    )
    module = instantiate(config.module, _convert_="all")
    configured = module.configure_optimizers()
    assert isinstance(configured, dict)
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert "betas" not in optimizer.defaults


def test_hydra_replaces_an_independent_model_stage_and_records_its_recipe() -> None:
    """A config-only experiment can replace a stage through its public interface."""
    config = _compose(
        [
            "~module.training_model.model.assembler",
            "+module.training_model.model.assembler._target_=tests.test_training_interfaces.ToyAssembler",
            "+module.training_model.model.assembler.variant=experiment",
        ]
    )
    module = instantiate(config.module, _convert_="all")

    assert isinstance(module.training_model.model.assembler, ToyAssembler)
    module.eval()
    with torch.no_grad():
        output = module(torch.zeros(1, 3, 64, 64))
    assert output.final.logits.shape == (1, 2, 3)
    recipe = module.training_model.checkpoint_config()
    assert recipe["model"]["config"]["assembler"] == {
        "target": "tests.test_training_interfaces.ToyAssembler",
        "config": {"variant": "experiment"},
    }


def test_dimension_overrides_reach_model_and_synthetic_data() -> None:
    """Shared dimensions keep every model stage and generated label compatible."""
    hidden_dim = 16
    num_classes = 2
    config = _compose(
        [f"dimensions.hidden_dim={hidden_dim}", f"dimensions.num_classes={num_classes}"]
    )
    module = instantiate(config.module, _convert_="all")
    dataset_factory = instantiate(config.datamodule.dataset_factory, _convert_="all")
    dataset = dataset_factory()
    model = module.training_model.model

    assert model.encoder.hidden_dim == hidden_dim
    assert model.projector.hidden_dim == hidden_dim
    assert model.selector.config.hidden_dim == hidden_dim
    assert model.decoder.config.hidden_dim == hidden_dim
    assert model.selector.config.num_classes == num_classes
    assert model.decoder.config.num_classes == num_classes
    assert module.training_model.denoising.hidden_dim == hidden_dim
    assert module.training_model.denoising.num_classes == num_classes
    labels = [target.labels for _, target in dataset if target.labels.numel()]
    assert labels
    assert max(int(label.max()) for label in labels) < num_classes


def _toy_di_config(*, reverse: bool) -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "clean_rtdetr.training.lightning.RTDETRLightningModule",
            "training_model": {
                "_target_": "tests.test_training_interfaces.ToyTrainingModel",
            },
            "criterion": {
                "_target_": "tests.test_training_interfaces.ToyCriterion",
                "matcher": {
                    "_target_": "tests.test_training_interfaces.FixedMatcher",
                    "reverse": reverse,
                },
            },
            "optimizer_factory": {
                "_target_": "torch.optim.SGD",
                "_partial_": True,
                "lr": 0.1,
            },
        }
    )


def test_config_only_nested_interfaces_train_and_validate_matcher_resume(tmp_path: Path) -> None:
    """Hydra selects arbitrary model, criterion, and matcher implementations recursively."""
    module = instantiate(_toy_di_config(reverse=False), _convert_="all")
    assert isinstance(module.training_model, ToyTrainingModel)
    assert isinstance(module.criterion, ToyCriterion)
    assert isinstance(module.criterion.matcher, FixedMatcher)
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", save_last=True)
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(module, train_dataloaders=_loader())
    assert checkpoint.last_model_path
    assert module.criterion.matcher.call_count.item() == 1

    incompatible = instantiate(_toy_di_config(reverse=True), _convert_="all")
    resume_trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=2,
        default_root_dir=tmp_path / "resume",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    with pytest.raises(ValueError, match="recipe"):
        resume_trainer.fit(
            incompatible,
            train_dataloaders=_loader(),
            ckpt_path=checkpoint.last_model_path,
        )
