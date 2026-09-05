# Copyright 2026 Clean RT-DETR contributors
"""Regression checks for the public Hydra training composition."""

from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from clean_rtdetr import RTDETR
from clean_rtdetr.training.lightning import RTDETRLightningModule


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "configs")


def _compose(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=_config_dir()):
        return compose(config_name="lightning_training", overrides=overrides or [])


def _targets(value: Any):  # noqa: ANN202, ANN401 - recursive test helper
    if hasattr(value, "items"):
        for key, child in value.items():
            if key == "_target_":
                yield str(child)
            yield from _targets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _targets(child)


def test_lightning_config_composes_full_model_without_example_targets() -> None:
    """The composed object graph builds the package model directly."""
    config = _compose()
    targets = tuple(_targets(OmegaConf.to_container(config, resolve=True)))
    assert targets
    assert all(not target.startswith("examples.") for target in targets)
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


def test_dimension_overrides_reach_model_and_synthetic_data() -> None:
    """Shared dimensions keep every model stage and generated label compatible."""
    hidden_dim = 16
    num_classes = 2
    config = _compose(
        [f"dimensions.hidden_dim={hidden_dim}", f"dimensions.num_classes={num_classes}"]
    )
    module = instantiate(config.module, _convert_="all")
    dataset = instantiate(config.dataset, _convert_="all")
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
