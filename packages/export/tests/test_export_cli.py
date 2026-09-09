# Copyright 2026 Clean RT-DETR contributors
"""Workflow tests for the fixed-shape ONNX export example CLI."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING

import pytest
from export_model_fixtures import small_model

from clean_rtdetr.checkpoint import CheckpointLoadReport
from clean_rtdetr_export import ExportVerificationError, cli

if TYPE_CHECKING:
    from pathlib import Path


def _small_model() -> object:
    return small_model()


def _load_report(*_args: object, **_kwargs: object) -> CheckpointLoadReport:
    return _loader_report()


def _no_threads(_count: int) -> None:
    return None


def _arguments(
    checkpoint: Path, output: Path, report: Path | None = None, *, verify_digest: bool = True
) -> list[str]:
    arguments = [
        "export_onnx.py",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if verify_digest:
        arguments.extend(("--expected-sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    if report is not None:
        arguments.extend(("--report", str(report)))
    return arguments


def _loader_report() -> CheckpointLoadReport:
    return CheckpointLoadReport("checkpoint-digest", ("a", "b"))


@pytest.mark.parametrize("verify_digest", [False, True])
def test_cli_success_writes_nested_report_and_prints_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    verify_digest: bool,
) -> None:
    """Successful export has one stable export object and compact load provenance."""
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"local checkpoint")
    output = tmp_path / "artifacts" / "model.onnx"
    report = tmp_path / "reports" / "model.json"
    monkeypatch.setattr(cli, "rtdetr_r50vd", _small_model)
    monkeypatch.setattr(cli, "load_model_weights", _load_report)

    def fake_export(_model: object, destination: Path) -> dict[str, object]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"onnx")
        return {"passed": True, "schema_version": 1}

    monkeypatch.setattr(cli, "export_onnx", fake_export)
    monkeypatch.setattr(cli.torch, "set_num_threads", _no_threads)
    monkeypatch.setattr(cli.torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(
        sys, "argv", _arguments(checkpoint, output, report, verify_digest=verify_digest)
    )

    cli.main()

    assert capsys.readouterr().out.strip() == str(report)
    payload = json.loads(report.read_text())
    assert payload["export"] == {"passed": True, "schema_version": 1}
    assert payload["checkpoint_load"] == {
        "format": "state_dict",
        "sha256": "checkpoint-digest",
        "tensor_count": 2,
    }
    assert payload["implementation_sha256"]
    assert payload["cli_sha256"]
    assert payload["checkpoint"]["expected_sha256"] == (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() if verify_digest else None
    )


def test_cli_failure_persists_export_report_and_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parity failure still writes evidence before returning a nonzero exit."""
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"local checkpoint")
    output = tmp_path / "model.onnx"
    report = tmp_path / "reports" / "failure.json"
    failure: dict[str, object] = {"passed": False, "checks": [{"name": "logits"}]}
    monkeypatch.setattr(cli, "rtdetr_r50vd", _small_model)
    monkeypatch.setattr(cli, "load_model_weights", _load_report)

    def fail_export(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ExportVerificationError(failure)

    monkeypatch.setattr(cli, "export_onnx", fail_export)
    monkeypatch.setattr(cli.torch, "set_num_threads", _no_threads)
    monkeypatch.setattr(cli.torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(sys, "argv", _arguments(checkpoint, output, report))

    with pytest.raises(SystemExit, match="ONNX output parity failed; see report"):
        cli.main()

    payload = json.loads(report.read_text())
    assert payload["export"] == failure
    assert payload["error"]


@pytest.mark.parametrize("collision", ["existing", "checkpoint"])
def test_cli_report_path_guards_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    """Existing or colliding report paths cannot overwrite input artifacts."""
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"local checkpoint")
    output = tmp_path / "model.onnx"
    report = tmp_path / "existing.json" if collision == "existing" else checkpoint
    if collision == "existing":
        report.write_text("sentinel")
    loaded = False

    def load(*_args: object, **_kwargs: object) -> CheckpointLoadReport:
        nonlocal loaded
        loaded = True
        return _loader_report()

    monkeypatch.setattr(cli, "load_model_weights", load)
    monkeypatch.setattr(sys, "argv", _arguments(checkpoint, output, report))
    with pytest.raises((FileExistsError, ValueError)):
        cli.main()
    assert not loaded
    if collision == "existing":
        assert report.read_text() == "sentinel"
