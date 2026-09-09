# Copyright 2026 Clean RT-DETR contributors
"""Evaluate a local COCO prediction JSON with pycocotools."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from clean_rtdetr.evaluation import evaluate_coco, load_coco_images


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source_provenance() -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[1]
    source = {str(path): _hash(path) for path in sorted(source_root.rglob("*.py"))}
    lockfile = Path("uv.lock")
    return {
        "implementation_sha256": source,
        "cli_sha256": _hash(Path(__file__).resolve()),
        "uv_lock_sha256": _hash(lockfile) if lockfile.is_file() else "unavailable",
    }


def _selected_ids(prediction_report: Path, annotations: Path, predictions: Path) -> list[int]:
    """Verify that a manifest describes these exact inputs before using its subset."""
    raw_report = json.loads(prediction_report.read_text())
    if not isinstance(raw_report, dict):
        message = "prediction report JSON must be an object"
        raise TypeError(message)
    recorded = raw_report.get("predictions")
    if not isinstance(recorded, dict) or recorded.get("sha256") != _hash(predictions):
        message = "predictions hash does not match prediction report"
        raise ValueError(message)
    recorded_annotations = raw_report.get("annotations")
    if not isinstance(recorded_annotations, dict) or recorded_annotations.get("sha256") != _hash(
        annotations
    ):
        message = "annotations hash does not match prediction report"
        raise ValueError(message)
    input_images = raw_report.get("input_images")
    if not isinstance(input_images, dict) or not isinstance(input_images.get("selected_ids"), list):
        message = "prediction report has no selected image IDs"
        raise TypeError(message)
    raw_ids = input_images["selected_ids"]
    if any(type(item) is not int or item < 0 for item in raw_ids):
        message = "prediction report selected IDs must be nonnegative integers"
        raise TypeError(message)
    return raw_ids


@dataclass(frozen=True)
class _EvaluationCommandArgs:
    annotations: Path
    predictions: Path
    prediction_report: Path | None
    report: Path


def run(args: Any) -> dict[str, object]:  # noqa: ANN401
    """Validate local predictions and write metric evidence without running inference."""
    annotations = args.annotations.resolve()
    predictions = args.predictions.resolve()
    report = args.report.resolve()
    if report.exists():
        message = f"report already exists: {report}"
        raise FileExistsError(message)
    prediction_report = args.prediction_report.resolve() if args.prediction_report else None
    if report in {annotations, predictions} or prediction_report == report:
        message = "report path must differ from input paths"
        raise ValueError(message)
    if not annotations.is_file() or not predictions.is_file():
        message = "annotations and predictions paths must exist"
        raise FileNotFoundError(message)
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(predictions.read_text())
    if not isinstance(payload, list):
        message = "predictions JSON must contain a list"
        raise TypeError(message)
    annotation_ids = [record.image_id for record in load_coco_images(annotations)]
    selected_ids = (
        _selected_ids(prediction_report, annotations, predictions)
        if prediction_report is not None
        else annotation_ids
    )
    scope = "all_annotation_images" if sorted(selected_ids) == annotation_ids else "subset"
    metrics = evaluate_coco(annotations, payload, image_ids=selected_ids)
    result = {
        "schema_version": 1,
        "metrics": metrics,
        "metric_scale": "0_to_100",
        "scope": scope,
        "image_ids": selected_ids,
        "annotations": {"path": str(annotations), "sha256": _hash(annotations)},
        "predictions": {
            "path": str(predictions),
            "sha256": _hash(predictions),
            "count": len(payload),
        },
        "prediction_report": (
            {"path": str(prediction_report), "sha256": _hash(prediction_report)}
            if prediction_report is not None
            else None
        ),
        "pycocotools": importlib.metadata.version("pycocotools"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **_source_provenance(),
    }
    _write(report, result)
    return result


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def evaluate(
    annotations: Annotated[Path, typer.Option(help="COCO annotation JSON.")],
    predictions: Annotated[Path, typer.Option(help="Prediction JSON.")],
    report: Annotated[Path, typer.Option(help="Metric report output path.")],
    prediction_report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Parse inputs for the optional official metric calculation."""
    args = _EvaluationCommandArgs(
        annotations=annotations,
        predictions=predictions,
        prediction_report=prediction_report,
        report=report,
    )
    print(json.dumps(run(args)["metrics"], sort_keys=True))


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
