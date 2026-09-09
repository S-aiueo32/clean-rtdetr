# Copyright 2026 Clean RT-DETR contributors
"""Export a verified, fixed-shape R50vd model to ONNX.

The command requires a locally available checkpoint. An expected digest can
optionally be supplied for verification. It never downloads model artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Annotated

import torch
import typer

import clean_rtdetr
from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr.checkpoint import CheckpointLoadReport, load_model_weights
from clean_rtdetr_export import ExportVerificationError, export_onnx


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*args: str, fallback: str) -> str:
    try:
        return subprocess.check_output(("git", *args), text=True).strip()  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _provenance(checkpoint: Path, expected_sha256: str | None) -> dict[str, object]:
    """Collect reproducibility metadata for the export report."""
    source_roots = (Path(clean_rtdetr.__file__).resolve().parent, Path(__file__).resolve().parent)
    implementation = {
        str(path): _sha256(path)
        for source_root in source_roots
        for path in sorted(source_root.rglob("*.py"))
    }
    lockfile = Path("uv.lock")
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "expected_sha256": expected_sha256,
            "format": "state_dict",
        },
        "git_revision": _git_output("rev-parse", "HEAD", fallback="unknown"),
        "worktree_dirty": bool(_git_output("status", "--porcelain", fallback="unknown")),
        "git_diff_sha256": hashlib.sha256(
            _git_output("diff", "HEAD", fallback="").encode()
        ).hexdigest(),
        "implementation_sha256": implementation,
        "cli_sha256": _sha256(Path(__file__).resolve()),
        "uv_lock_sha256": _sha256(lockfile) if lockfile.is_file() else "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
    }


def _load_report(report: CheckpointLoadReport) -> dict[str, object]:
    """Convert the checkpoint loader's dataclass report to JSON data."""
    return {
        "sha256": report.sha256,
        "format": "state_dict",
        "tensor_count": len(report.keys),
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def export(
    checkpoint: Annotated[Path, typer.Option(help="Native state_dict checkpoint path.")],
    output: Annotated[Path, typer.Option(help="ONNX output path.")],
    expected_sha256: Annotated[
        str | None, typer.Option(help="Optional expected checkpoint SHA-256 to verify.")
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Optional report path.")] = None,
) -> None:
    """Load weights, export ONNX, verify runtime parity, and write a report."""
    report_path = report or output.with_suffix(".json")
    if report_path.exists():
        message = f"Report already exists: {report_path}"
        raise FileExistsError(message)
    report_resolved = report_path.resolve()
    if report_resolved in {output.resolve(), checkpoint.resolve()}:
        message = "Report path must differ from output and checkpoint paths"
        raise ValueError(message)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    model = rtdetr_r50vd().eval().cpu()
    loaded = load_model_weights(checkpoint, model=model, expected_sha256=expected_sha256)
    metadata = _provenance(checkpoint, expected_sha256)
    report_base: dict[str, object] = {
        "checkpoint_load": _load_report(loaded),
        "output": str(output),
        **metadata,
    }
    try:
        export_report = export_onnx(model, output)
    except ExportVerificationError as error:
        report_base["export"] = error.report
        report_base["error"] = str(error)
        _write_report(report_path, report_base)
        message = "ONNX output parity failed; see report"
        raise SystemExit(message) from error
    report_base["export"] = export_report
    _write_report(report_path, report_base)
    print(report_path)


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
