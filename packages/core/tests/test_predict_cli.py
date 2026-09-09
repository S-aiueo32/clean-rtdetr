# Copyright 2026 Clean RT-DETR contributors
"""Verify stdout filtering and optional image rendering through the prediction CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageChops
from torch import nn
from typer.testing import CliRunner

from clean_rtdetr.cli import predict as prediction_cli
from clean_rtdetr.core.contracts import DetectionOutput

NUM_DETECTIONS = 300
RUNNER = CliRunner()


class FakeModel(nn.Module):
    """Emit ranked candidates with an exact threshold boundary and off-canvas boxes."""

    def forward(self, images: torch.Tensor, *, plan: object = None) -> SimpleNamespace:
        """Exercise real preprocessing and postprocessing around a cheap model."""
        del plan
        batch = images.shape[0]
        logits = torch.full((batch, NUM_DETECTIONS, 80), -20.0)
        logits[:, 0, 15] = 2.0
        logits[:, 1, 16] = 0.0
        logits[:, 2, 0] = -2.0
        boxes = torch.full((batch, NUM_DETECTIONS, 4), 0.5)
        boxes[:, 0] = torch.tensor([0.5, 0.5, 1.5, 1.5])
        return SimpleNamespace(final=DetectionOutput(logits, boxes))


def _loader(*_args: object, **_kwargs: object) -> None:
    """Avoid a large weight fixture while retaining the real CLI control flow."""


@pytest.fixture
def sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an RGB input and install the deterministic inference stand-in."""
    path = tmp_path / "cat.png"
    Image.new("RGB", (100, 80), "white").save(path)
    monkeypatch.setattr(prediction_cli, "rtdetr_r50vd", FakeModel)
    monkeypatch.setattr(prediction_cli, "load_model_weights", _loader)
    return path


@pytest.mark.parametrize(("threshold", "count"), [(None, NUM_DETECTIONS), ("0.5", 2), ("1", 0)])
def test_stdout_threshold_preserves_aligned_predictions(
    sample: Path, threshold: str | None, count: int
) -> None:
    """Defaults keep 300 candidates; inclusive filtering retains aligned, unclipped boxes."""
    command = [str(sample), "--checkpoint", "unused.pth"]
    if threshold is not None:
        command.extend(["--conf-thresh", threshold])
    result = RUNNER.invoke(prediction_cli.app, command)
    assert result.exit_code == 0, result.exception
    rows = json.loads(result.output)
    assert len(rows) == 1
    row = rows[0]
    assert row["image"] == str(sample)
    assert row["original_size_wh"] == [100, 80]
    assert len(row["labels"]) == len(row["scores"]) == len(row["boxes_xyxy"]) == count
    if count:
        assert row["labels"][:2] == [15, 16]
        assert row["scores"][:2] == pytest.approx([torch.sigmoid(torch.tensor(2.0)).item(), 0.5])
        assert row["boxes_xyxy"][:2] == [[-25.0, -20.0, 125.0, 100.0], [25.0, 20.0, 75.0, 60.0]]
    assert sorted(path.name for path in sample.parent.iterdir()) == ["cat.png"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_threshold_fails_before_loading(threshold: float) -> None:
    """Reject invalid thresholds before touching nonexistent input or weight files."""
    with pytest.raises(ValueError, match="conf_thresh"):
        prediction_cli.run([Path("missing.png")], Path("missing.pth"), conf_thresh=threshold)


def test_cli_rejects_invalid_threshold() -> None:
    """Surface invalid confidence values at the command-line boundary."""
    result = RUNNER.invoke(
        prediction_cli.app,
        ["missing.png", "--checkpoint", "missing.pth", "--conf-thresh", "nan"],
    )
    assert result.exit_code != 0
    assert "conf" in str(result.exception).lower() or "conf" in result.output.lower()


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_visualizations_preserve_inputs_and_handle_empty_results(
    sample: Path, tmp_path: Path, threshold: float
) -> None:
    """Save distinct PNGs for same-named inputs, including an unannotated empty result."""
    second = tmp_path / "other" / sample.name
    second.parent.mkdir()
    Image.new("RGB", (120, 90), "white").save(second)
    before = [path.read_bytes() for path in (sample, second)]
    destination = tmp_path / "nested" / "visualizations"
    result = RUNNER.invoke(
        prediction_cli.app,
        [
            str(sample),
            str(second),
            "--checkpoint",
            "unused.pth",
            "--conf-thresh",
            str(threshold),
            "--visualize-dir",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.exception
    rows = json.loads(result.output)
    assert len(rows) == len(before)
    assert [path.read_bytes() for path in (sample, second)] == before
    assert sorted(path.name for path in destination.iterdir()) == ["0001-cat.png", "0002-cat.png"]
    for source, name in zip((sample, second), ("0001-cat.png", "0002-cat.png"), strict=True):
        with Image.open(source) as original, Image.open(destination / name) as rendered:
            assert rendered.format == "PNG"
            assert rendered.size == original.size
            difference = ImageChops.difference(original, rendered).getbbox()
            assert (difference is None) == (threshold == 1.0)


def test_visualization_cannot_overwrite_another_input(tmp_path: Path) -> None:
    """Validate all destinations before overwriting an input from the same batch."""
    first = tmp_path / "cat.png"
    second = tmp_path / "0001-cat.png"
    for path in (first, second):
        Image.new("RGB", (40, 30), "white").save(path)
    before = second.read_bytes()
    with pytest.raises(ValueError, match="overwrit"):
        prediction_cli.run([first, second], Path("missing.pth"), visualize_dir=tmp_path)
    assert second.read_bytes() == before
