# Copyright 2026 Clean RT-DETR contributors
"""Hydra composition helpers for caller-owned training recipes."""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_training_config(
    name: str, overrides: Sequence[str] = (), *, config_dir: str | Path
) -> DictConfig:
    """Compose a recipe from an explicit directory, resolving relative paths from cwd."""
    directory = Path(config_dir).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        return compose(config_name=name, overrides=list(overrides))
