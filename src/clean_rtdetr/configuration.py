# Copyright 2026 Clean RT-DETR contributors
"""Hydra composition helpers for packaged model recipes."""

from collections.abc import Sequence
from importlib.resources import as_file, files

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_config(name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose a packaged recipe independently of the working directory."""
    resource = files("clean_rtdetr").joinpath("configs")
    with (
        as_file(resource) as config_dir,
        initialize_config_dir(version_base="1.3", config_dir=str(config_dir)),
    ):
        return compose(config_name=name, overrides=list(overrides))


def compose_model_config(name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose a packaged model recipe and return its model node."""
    return compose_config(name, overrides).model
