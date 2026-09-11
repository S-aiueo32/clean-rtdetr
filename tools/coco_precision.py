# Copyright 2026 Clean RT-DETR contributors
"""Compare CUDA FP32/FP16-autocast COCO metrics with identical inference I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

from clean_rtdetr.cli import coco_evaluate, coco_predict
from clean_rtdetr.evaluation import load_coco_images


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Evaluate both precisions, requiring complete and matching image coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=5000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; CPU execution cannot validate CUDA autocast")
    records = load_coco_images(args.annotations)
    if len(records) != args.expected_images or args.expected_images < 1:
        parser.error("annotation image count does not match --expected-images")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = {}
    prediction_reports = {}
    for precision in ("fp32", "amp_fp16"):
        directory = args.output / precision
        predictions = directory / "predictions.json"
        prediction_report = directory / "predictions.report.json"
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "amp_fp16"):
            report = coco_predict.run(
                SimpleNamespace(
                    images=args.images,
                    annotations=args.annotations,
                    checkpoint=args.checkpoint,
                    expected_sha256=args.expected_sha256,
                    output=predictions,
                    report=prediction_report,
                    limit=None,
                    device="cuda",
                )
            )
        report["execution_precision"] = precision
        report["weights_dtype"] = "float32"
        report["public_prediction_dtype"] = "float32"
        report["tf32"] = False
        _write(prediction_report, report)
        metrics = coco_evaluate.run(
            SimpleNamespace(
                annotations=args.annotations,
                predictions=predictions,
                prediction_report=prediction_report,
                report=directory / "metrics.json",
            )
        )
        if report["scope"] != "all_annotation_images" or metrics["scope"] != report["scope"]:
            message = "evaluation did not cover every annotation image"
            raise RuntimeError(message)
        results[precision] = metrics
        prediction_reports[precision] = report
        torch.cuda.empty_cache()
    fp32_images = cast("dict[str, object]", prediction_reports["fp32"]["input_images"])
    amp_images = prediction_reports["amp_fp16"]["input_images"]
    if fp32_images != amp_images:
        message = "precision runs used different input image bytes or IDs"
        raise RuntimeError(message)
    fp32_metrics = cast("dict[str, float]", results["fp32"]["metrics"])
    amp_metrics = cast("dict[str, float]", results["amp_fp16"]["metrics"])
    metric_delta = {name: amp_metrics[name] - value for name, value in fp32_metrics.items()}
    summary = {
        "status": "completed",
        "scope": "all_annotation_images",
        "seed": 41,
        "image_count": len(records),
        "ordered_image_digest": fp32_images["ordered_digest"],
        "checkpoint_sha256": args.expected_sha256,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "tf32": False,
        },
        "metrics": {name: value["metrics"] for name, value in results.items()},
        "amp_minus_fp32_percentage_points": metric_delta,
        "artifact_sha256": {
            str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.rglob("*.json"))
        },
        "note": "Measured precision differences; no universal AP degradation threshold assumed.",
    }
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
