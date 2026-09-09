# Copyright 2026 Clean RT-DETR contributors
"""Native checkpoint validation and official conversion integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from core_model_fixtures import small_model
from test_checkpoint import _full_state
from torch import nn
from typer.testing import CliRunner

from clean_rtdetr import DecoderPlan
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.cli import convert as convert_checkpoint


def _save(path: Path, payload: object) -> str:
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converter_defaults_match_official_provenance() -> None:
    """Keep installed conversion defaults aligned with the pinned parity artifact."""
    baseline = Path(__file__).resolve().parents[3] / "benchmarks/baseline.json"
    checkpoint = json.loads(baseline.read_text())["checkpoint"]
    assert checkpoint["source"] == convert_checkpoint.DEFAULT_SOURCE
    assert checkpoint["sha256"] == convert_checkpoint.DEFAULT_SHA256


@pytest.mark.parametrize("verify_digest", [False, True])
def test_native_checkpoint_round_trip(tmp_path: Path, *, verify_digest: bool) -> None:
    """A plain PyTorch state_dict restores parameters and buffers exactly."""
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).eval()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "weights.pth"
    digest = _save(path, model.state_dict())
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.zero_()
    report = (
        load_model_weights(path, model=model, expected_sha256=digest)
        if verify_digest
        else load_model_weights(path, model=model)
    )
    assert report.sha256 == digest
    assert report.keys == tuple(expected)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "failure", ["hash", "model", "ema", "missing", "extra", "shape", "dtype", "meta", "sparse"]
)
def test_native_validation_precedes_parameter_changes(tmp_path: Path, failure: str) -> None:
    """Reject envelopes and invalid late tensors without changing earlier weights."""
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value + 1 for key, value in before.items()}
    payload: object = state
    if failure == "model":
        payload = {"model": state}
    elif failure == "ema":
        payload = {"ema": {"module": state}}
    elif failure == "missing":
        del state["1.bias"]
    elif failure == "extra":
        state["extra"] = torch.zeros(1)
    elif failure != "hash":
        state["1.bias"] = {
            "shape": lambda: torch.zeros(3),
            "dtype": state["1.bias"].double,
            "meta": lambda: state["1.bias"].to("meta"),
            "sparse": state["1.bias"].to_sparse,
        }[failure]()
    path = tmp_path / "weights.pth"
    _save(path, payload)
    with pytest.raises(ValueError, match="Checkpoint"):
        load_model_weights(
            path, model=model, expected_sha256="incorrect" if failure == "hash" else None
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("source", ["model", "ema"])
def test_official_conversion_preserves_native_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Convert an official envelope and reproduce predictions with the native loader."""
    model = small_model().eval()
    state = _full_state(model)
    official = tmp_path / "official.pth"
    digest = _save(official, {source: state if source == "model" else {"module": state}})
    output = tmp_path / "converted" / "weights.pth"
    monkeypatch.setattr(convert_checkpoint, "rtdetr_r50vd", lambda: model)
    result = CliRunner().invoke(
        convert_checkpoint.app,
        [
            "--checkpoint",
            str(official),
            *(["--source", source] if source != "ema" else []),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    report = json.loads(result.output)
    assert report["official_checkpoint_load"]["source"] == source
    images = torch.rand(1, 3, 64, 64)
    plan = DecoderPlan(collect_intermediate=False)
    with torch.inference_mode():
        expected = model(images, plan=plan).final
        for tensor in model.state_dict().values():
            tensor.zero_()
        loaded = load_model_weights(output, model=model, expected_sha256=report["sha256"])
        actual = model(images, plan=plan).final
    assert loaded.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.boxes, expected.boxes, rtol=0, atol=0)


def test_converter_preserves_existing_output(tmp_path: Path) -> None:
    """Conversion refuses to overwrite a previously produced checkpoint."""
    output = tmp_path / "weights.pth"
    output.write_bytes(b"keep")
    result = CliRunner().invoke(
        convert_checkpoint.app,
        [
            "--checkpoint",
            str(output),
            "--source",
            "model",
            "--expected-sha256",
            "unused",
            "--output",
            str(output),
        ],
    )
    assert isinstance(result.exception, FileExistsError)
    assert output.read_bytes() == b"keep"


def test_converter_default_digest_rejects_unrecognized_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting the expected hash still verifies the pinned official artifact."""
    path = tmp_path / "unknown.pth"
    path.write_bytes(b"not the pinned checkpoint")
    output = tmp_path / "native.pth"
    monkeypatch.setattr(convert_checkpoint, "rtdetr_r50vd", small_model)
    result = CliRunner().invoke(
        convert_checkpoint.app, ["--checkpoint", str(path), "--output", str(output)]
    )
    assert isinstance(result.exception, ValueError)
    assert "SHA-256 mismatch" in str(result.exception)
    assert not output.exists()
