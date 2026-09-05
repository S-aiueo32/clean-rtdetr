# Copyright 2026 Clean RT-DETR contributors
"""Export a verified, fixed-shape R50vd model to ONNX.

The command deliberately requires a locally available checkpoint and its
expected digest.  It never downloads model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Literal, cast

import torch

from clean_rtdetr import build_rtdetr_r50vd
from clean_rtdetr.compatibility import CheckpointLoadReport, load_official_model_weights
from clean_rtdetr.export import ExportVerificationError, export_onnx


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*args: str, fallback: str) -> str:
    try:
        return subprocess.check_output(("git", *args), text=True).strip()  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _provenance(checkpoint: Path, expected_sha256: str, source: str) -> dict[str, object]:
    """Collect reproducibility metadata for the export report."""
    source_root = Path("src/clean_rtdetr")
    implementation = {str(path): _sha256(path) for path in sorted(source_root.rglob("*.py"))}
    lockfile = Path("uv.lock")
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "expected_sha256": expected_sha256,
            "source": source,
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
    """Convert the compatibility loader's dataclass report to JSON data."""
    return {
        "sha256": report.sha256,
        "source": report.source,
        "mapped_key_count": len(report.key_mapping),
        "excluded_keys": report.excluded_keys,
        "synthesized_keys": list(report.synthesized_keys),
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Load weights, export ONNX, verify runtime parity, and write a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", choices=("model", "ema"), required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".json")
    if report_path.exists():
        message = f"Report already exists: {report_path}"
        raise FileExistsError(message)
    report_resolved = report_path.resolve()
    if report_resolved in {args.output.resolve(), args.checkpoint.resolve()}:
        message = "Report path must differ from output and checkpoint paths"
        raise ValueError(message)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    model = build_rtdetr_r50vd().eval().cpu()
    loaded = load_official_model_weights(
        args.checkpoint,
        model=model,
        source=cast("Literal['model', 'ema']", args.source),
        expected_sha256=args.expected_sha256,
    )
    metadata = _provenance(args.checkpoint, args.expected_sha256, args.source)
    report_base: dict[str, object] = {
        "checkpoint_load": _load_report(loaded),
        "output": str(args.output),
        **metadata,
    }
    try:
        export_report = export_onnx(model, args.output)
    except ExportVerificationError as error:
        report_base["export"] = error.report
        report_base["error"] = str(error)
        _write_report(report_path, report_base)
        message = "ONNX output parity failed; see report"
        raise SystemExit(message) from error
    report_base["export"] = export_report
    _write_report(report_path, report_base)
    print(report_path)


if __name__ == "__main__":
    main()
