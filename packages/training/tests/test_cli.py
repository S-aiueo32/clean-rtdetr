# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the training command-line boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from typer.testing import CliRunner

import clean_rtdetr_training.cli as training_cli

if TYPE_CHECKING:
    from omegaconf import DictConfig


@pytest.mark.parametrize("args", [[], ["--config-dir", "."], ["--config-name", "recipe"]])
def test_train_cli_requires_explicit_recipe(args: list[str]) -> None:
    """The installed command requires a caller-owned configuration location and name."""
    result = CliRunner().invoke(training_cli.app, args)
    assert result.exit_code == 2  # noqa: PLR2004 - Click usage error
    assert "Missing option" in result.output


def test_train_cli_composes_external_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A custom recipe outside the checkout accepts positional Hydra overrides."""
    config_dir = tmp_path / "experiment"
    config_dir.mkdir()
    (config_dir / "custom.yaml").write_text("trainer:\n  max_epochs: 2\n")
    configs: list[DictConfig] = []
    monkeypatch.setattr(training_cli, "run", configs.append)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        training_cli.app,
        ["--config-dir", "experiment", "--config-name", "custom", "trainer.max_epochs=1"],
    )

    assert result.exit_code == 0, result.output
    assert len(configs) == 1
    assert configs[0].trainer.max_epochs == 1


def test_train_cli_builds_nested_trainer_callbacks(tmp_path: Path) -> None:
    """Nested callbacks train with the shared data RNG and save their checkpoint state."""
    result = CliRunner().invoke(
        training_cli.app,
        [
            "--config-dir",
            str(Path(__file__).resolve().parents[3] / "configs"),
            "--config-name",
            "lightning_training",
            "trainer.max_epochs=1",
            f"output_dir={tmp_path}",
        ],
    )

    assert result.exit_code == 0, result.output
    checkpoint = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert "EpochBoundaryState" in checkpoint["callbacks"]
    assert "RTDETREMA" in checkpoint["callbacks"]
