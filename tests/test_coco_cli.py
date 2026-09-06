# Copyright 2026 Clean RT-DETR contributors
"""Workflow tests for the local COCO prediction and metric CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from PIL import Image
from torch import nn

import clean_rtdetr.cli.coco_evaluate as evaluation_cli
import clean_rtdetr.cli.coco_predict as predictions_cli
from clean_rtdetr.compatibility import CheckpointLoadReport
from clean_rtdetr.core.contracts import DetectionOutput
from clean_rtdetr.evaluation import coco_predictions
from tests.test_coco import annotation_data, detections, write_annotations

DECODER_LAYERS = 6


class FakeModel(nn.Module):
    """Small inference stand-in exposing the model output contract."""

    def forward(self, images: torch.Tensor, *, plan: object = None) -> SimpleNamespace:
        """Return finite zero logits and centered boxes for each image."""
        del plan
        batch = images.shape[0]
        return SimpleNamespace(
            final=DetectionOutput(
                torch.zeros((batch, 300, 80)),
                torch.full((batch, 300, 4), 0.5),
            )
        )


def _args(**values: object) -> argparse.Namespace:
    defaults = {
        "images": Path("images"),
        "annotations": Path("annotations.json"),
        "checkpoint": Path("checkpoint.pth"),
        "source": "model",
        "expected_sha256": "unused",
        "output": Path("predictions.json"),
        "report": None,
        "limit": None,
        "device": "cpu",
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    annotations = write_annotations(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for name in ("one.jpg", "two.jpg"):
        Image.new("RGB", (100, 100), color="white").save(images / name)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    return annotations, images, checkpoint


def _loader(*_args: object, **_kwargs: object) -> CheckpointLoadReport:
    return CheckpointLoadReport("digest", "model", {}, {}, ())


def test_predictions_cli_writes_subset_json_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write a nested subset prediction file and compact manifest."""
    annotations, images, checkpoint = _prepare(tmp_path)
    output = tmp_path / "nested" / "predictions.json"
    report = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(predictions_cli, "instantiate", lambda _config, **_kwargs: FakeModel())
    monkeypatch.setattr(predictions_cli, "load_official_model_weights", _loader)
    args = _args(
        annotations=annotations,
        images=images,
        checkpoint=checkpoint,
        expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        output=output,
        report=report,
        limit=1,
    )
    result = predictions_cli.run(args)
    payload = json.loads(output.read_text())
    assert payload
    assert {item["image_id"] for item in payload} == {1}
    input_images = cast("dict[str, object]", result["input_images"])
    assert input_images["selected_ids"] == [1]
    assert result["schema_version"] == 1
    assert result["dtype"] == "float32"
    assert result["decoder_layers"] == DECODER_LAYERS


def test_predictions_cli_rejects_dimensions_and_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject files whose dimensions disagree or whose names escape the root."""
    annotations, images, checkpoint = _prepare(tmp_path)
    monkeypatch.setattr(predictions_cli, "instantiate", lambda _config, **_kwargs: FakeModel())
    monkeypatch.setattr(predictions_cli, "load_official_model_weights", _loader)
    Image.new("RGB", (99, 100), color="white").save(images / "one.jpg")
    args = _args(
        annotations=annotations,
        images=images,
        checkpoint=checkpoint,
        output=tmp_path / "results.json",
    )
    with pytest.raises(ValueError, match="dimensions"):
        predictions_cli.run(args)
    Image.new("RGB", (100, 100), color="white").save(images / "one.jpg")
    data = annotation_data()
    image_data = data["images"]
    assert isinstance(image_data, list)
    assert isinstance(image_data[0], dict)
    image_data[0]["file_name"] = "../outside.jpg"
    escaped = write_annotations(tmp_path, data)
    args.annotations = escaped
    with pytest.raises(ValueError, match="escapes"):
        predictions_cli.run(args)


@pytest.mark.parametrize("invalid", ["ids", "predictions", "annotations"])
def test_evaluation_cli_rejects_noninteger_report_ids_and_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    """Reject malformed prediction manifests before invoking the evaluator."""
    annotations = write_annotations(tmp_path)
    predictions = tmp_path / "predictions.json"
    predictions.write_text("[]")
    prediction_report = tmp_path / "prediction.report.json"
    prediction_report.write_text(
        json.dumps(
            {
                "predictions": {
                    "sha256": "wrong"
                    if invalid == "predictions"
                    else hashlib.sha256(predictions.read_bytes()).hexdigest()
                },
                "annotations": {
                    "sha256": "wrong"
                    if invalid == "annotations"
                    else hashlib.sha256(annotations.read_bytes()).hexdigest()
                },
                "input_images": {"selected_ids": [1.5] if invalid == "ids" else [1]},
            }
        )
    )

    def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, float | None]:
        pytest.fail("Invalid manifest reached the evaluator")

    monkeypatch.setattr(evaluation_cli, "evaluate_coco", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coco_evaluation.py",
            "--annotations",
            str(annotations),
            "--predictions",
            str(predictions),
            "--prediction-report",
            str(prediction_report),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    expected_error = TypeError if invalid == "ids" else ValueError
    match = "nonnegative integers" if invalid == "ids" else "hash does not match"
    with pytest.raises(expected_error, match=match):
        evaluation_cli.main()


@pytest.mark.parametrize("mode", ["full", "subset", "missing", "full_report"])
def test_evaluation_cli_keeps_full_and_subset_scope(tmp_path: Path, mode: str) -> None:
    """JSON evaluation preserves missed images unless a verified subset was requested."""
    pytest.importorskip("pycocotools")
    annotations = write_annotations(tmp_path)
    predictions = tmp_path / "predictions.json"
    rows = coco_predictions(detections(), torch.tensor([1, 2]))
    predictions.write_text(json.dumps(rows[:1] if mode in {"subset", "missing"} else rows))
    manifest = tmp_path / "prediction.report.json"
    manifest.write_text(
        json.dumps(
            {
                "predictions": {"sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()},
                "annotations": {"sha256": hashlib.sha256(annotations.read_bytes()).hexdigest()},
                "input_images": {"selected_ids": [1] if mode == "subset" else [2, 1]},
            }
        )
    )
    report = tmp_path / "metrics" / "result.json"
    result = evaluation_cli.run(
        argparse.Namespace(
            annotations=annotations,
            predictions=predictions,
            prediction_report=manifest if mode in {"subset", "full_report"} else None,
            report=report,
        )
    )
    saved = json.loads(report.read_text())
    assert saved == result
    assert saved["metric_scale"] == "0_to_100"
    assert saved["scope"] == ("subset" if mode == "subset" else "all_annotation_images")
    assert saved["metrics"]["AP"] == pytest.approx(50.0 if mode == "missing" else 100.0)
    assert saved["metrics"]["AP_large"] is None
