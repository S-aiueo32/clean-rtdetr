# Copyright 2026 Clean RT-DETR contributors
"""Hydra composition helpers for packaged training recipes."""

from collections.abc import Sequence
from importlib.resources import as_file, files

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_training_config(name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose a packaged training recipe independently of the working directory."""
    resource = files("clean_rtdetr_training").joinpath("configs")
    with (
        as_file(resource) as config_dir,
        initialize_config_dir(version_base="1.3", config_dir=str(config_dir)),
    ):
        return compose(config_name=name, overrides=list(overrides))
