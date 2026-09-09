# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the training command-line boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

import clean_rtdetr_training.cli as training_cli

if TYPE_CHECKING:
    import pytest


def test_train_cli_allows_bare_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare command composes the default recipe with no overrides."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        training_cli,
        "_compose_config",
        lambda overrides: calls.append(overrides) or SimpleNamespace(),
    )

    def fake_run(_config: object) -> None:
        return None

    monkeypatch.setattr(training_cli, "run", fake_run)

    result = CliRunner().invoke(training_cli.app, [])

    assert result.exit_code == 0
    assert calls == [[]]


def test_train_cli_forwards_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positional Hydra overrides reach the public composition helper."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        training_cli,
        "_compose_config",
        lambda overrides: calls.append(overrides) or SimpleNamespace(),
    )

    def fake_run(_config: object) -> None:
        return None

    monkeypatch.setattr(training_cli, "run", fake_run)

    result = CliRunner().invoke(
        training_cli.app,
        ["trainer.max_epochs=1", "output_dir=outputs/test"],
    )

    assert result.exit_code == 0
    assert calls == [["trainer.max_epochs=1", "output_dir=outputs/test"]]
