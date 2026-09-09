# Copyright 2026 Clean RT-DETR contributors
"""Regression checks for the public Hydra training composition."""

from pathlib import Path
from shutil import copytree
from typing import Any

import lightning
import pytest
import torch
from hydra.errors import MissingConfigException
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from test_training_interfaces import (
    FixedMatcher,
    ToyAssembler,
    ToyCriterion,
    ToyTrainingModel,
    _loader,
)

from clean_rtdetr import RTDETR, get_model_preset, list_models
from clean_rtdetr_training.configuration import compose_training_config
from clean_rtdetr_training.lightning import RTDETRLightningModule


def _compose(overrides: list[str] | None = None) -> DictConfig:
    return compose_training_config(
        "lightning_training",
        overrides or [],
        config_dir=Path(__file__).resolve().parents[3] / "configs",
    )


def _targets(value: Any):  # noqa: ANN202, ANN401 - recursive test helper
    if hasattr(value, "items"):
        for key, child in value.items():
            if key == "_target_":
                yield str(child)
            yield from _targets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _targets(child)


def test_external_config_tree_composes_away_from_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied recipes retain nested groups and permit caller-owned additions."""
    learning_rate = 0.25
    config_dir = tmp_path / "recipes"
    copytree(Path(__file__).resolve().parents[3] / "configs", config_dir)
    (config_dir / "module/optimizer_factory/custom.yaml").write_text(
        f"_target_: torch.optim.SGD\n_partial_: true\nlr: {learning_rate}\n"
    )
    monkeypatch.chdir(tmp_path)
    for directory in ("recipes", config_dir):
        config = compose_training_config(
            "lightning_training", ["module/optimizer_factory=custom"], config_dir=directory
        )
        assert config.module.optimizer_factory.lr == learning_rate
        assert config.datamodule._target_.endswith("DetectionDataModule.from_training_model")


def test_missing_external_config_does_not_fall_back_to_repository(tmp_path: Path) -> None:
    """Missing directories or recipes surface Hydra's configuration errors."""
    for directory in (tmp_path, tmp_path / "missing"):
        with pytest.raises(MissingConfigException):
            compose_training_config("lightning_training", config_dir=directory)


def test_lightning_config_composes_full_model_without_command_targets() -> None:
    """The composed object graph builds the package model directly."""
    config = _compose()
    targets = tuple(_targets(OmegaConf.to_container(config, resolve=True)))
    assert targets
    assert all(".cli." not in target for target in targets)
    module = instantiate(config.module, _convert_="all")
    assert isinstance(module, RTDETRLightningModule)
    assert isinstance(module.training_model.model, RTDETR)
    assert module.training_model.model.encoder.specs == module.training_model.model.backbone.specs


def test_lightning_config_group_overrides_build_optimizer_and_scheduler() -> None:
    """Config groups replace factories and bind them to the actual model parameters."""
    config = _compose(["module/optimizer_factory=sgd", "module/scheduler_factory=step"])
    module = instantiate(config.module, _convert_="all")
    configured = module.configure_optimizers()
    assert isinstance(configured, dict)
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert "betas" not in optimizer.defaults


def test_nested_optional_config_groups_can_be_removed() -> None:
    """Removing a child group removes the instantiated component from its parent."""
    config = _compose(["~module/scheduler_factory", "~trainer/callbacks/ema"])
    module = instantiate(config.module, _convert_="all")
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)
    assert "callbacks" not in config
    assert set(config.trainer.callbacks) == {"rng", "checkpoint"}


@pytest.mark.parametrize("name", list_models())
def test_named_model_config_builds_compatible_training_components(name: str) -> None:
    """Every factory preset composes with DN and sufficiently large synthetic images."""
    num_classes = 2
    num_queries = 300
    config = _compose(
        [
            f"module/training_model/model={name}",
            f"module.training_model.model.num_classes={num_classes}",
        ]
    )
    assert config.module.training_model.model._target_ == "clean_rtdetr.create_model"
    with torch.device("meta"):
        module = instantiate(config.module, _convert_="all")
    model = module.training_model.model
    preset = get_model_preset(name)
    assert isinstance(model, RTDETR)
    assert model.hidden_dim == preset.hidden_dim == module.training_model.denoising.hidden_dim
    assert model.num_classes == module.training_model.denoising.num_classes == num_classes
    assert "dimensions" not in config
    assert model.decoder.config.num_layers == preset.decoder_layers
    assert model.num_queries == num_queries
    dataset_factory = instantiate(config.datamodule.dataset_factory, _convert_="all")
    dataset = dataset_factory(training_model=module.training_model)
    assert dataset.num_classes == num_classes
    images, _ = dataset[0]
    height, width = images.shape[-2:]
    assert (
        sum((height // spec.stride) * (width // spec.stride) for spec in model.backbone.specs)
        >= num_queries
    )


@pytest.mark.parametrize("name", ["rtdetr_r18vd", "rtdetrv2_r18vd"])
def test_named_model_config_runs_training_forward(name: str) -> None:
    """Factory configs support real predictions with their default data and DN settings."""
    config = _compose([f"module/training_model/model={name}"])
    module = instantiate(config.module, _convert_="all")
    dataset_factory = instantiate(config.datamodule.dataset_factory, _convert_="all")
    images, target = dataset_factory(training_model=module.training_model)[0]
    with torch.no_grad():
        output = module.training_model(images.unsqueeze(0), targets=[target])
    assert output.final.logits.shape == (1, 300, module.training_model.model.num_classes)


def test_hydra_replaces_an_independent_model_stage_and_records_its_recipe() -> None:
    """A config-only experiment can replace a stage through its public interface."""
    config = _compose(
        [
            "~module.training_model.model.assembler",
            "+module.training_model.model.assembler._target_=test_training_interfaces.ToyAssembler",
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
        "target": "test_training_interfaces.ToyAssembler",
        "config": {"variant": "experiment"},
    }


def test_owned_dimensions_reach_model_and_delayed_factories() -> None:
    """DN and synthetic data consume the constructed model's dimensions."""
    hidden_dim = 16
    num_classes = 2
    config = _compose(
        [
            f"module.training_model.model.encoder.config.hidden_dim={hidden_dim}",
            f"module.training_model.model.selector.config.num_classes={num_classes}",
        ]
    )
    module = instantiate(config.module, _convert_="all")
    dataset_factory = instantiate(config.datamodule.dataset_factory, _convert_="all")
    dataset = dataset_factory(training_model=module.training_model)
    model = module.training_model.model

    assert dataset.num_classes == num_classes
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
            "_target_": "clean_rtdetr_training.lightning.RTDETRLightningModule",
            "training_model": {
                "_target_": "test_training_interfaces.ToyTrainingModel",
            },
            "criterion": {
                "_target_": "test_training_interfaces.ToyCriterion",
                "matcher": {
                    "_target_": "test_training_interfaces.FixedMatcher",
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
