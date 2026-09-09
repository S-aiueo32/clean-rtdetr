# Copyright 2026 Clean RT-DETR contributors
"""Contain the optional COCO API and its mutable objects at the metric boundary.

Only this module loads pycocotools. Its inputs and outputs use project-owned
Python data; no COCO objects, NumPy arrays, or native extension types escape.
"""

import copy
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from clean_rtdetr.evaluation.coco import CocoPrediction

METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def evaluate_bbox(
    annotations: Path, predictions: Sequence[CocoPrediction], image_ids: list[int]
) -> dict[str, float | None]:
    """Evaluate validated predictions using the official bbox defaults."""
    try:
        coco_class = import_module("pycocotools.coco").COCO
        evaluator_class = import_module("pycocotools.cocoeval").COCOeval
    except ImportError as error:
        msg = "AP calculation requires clean-rtdetr[evaluation]; prediction JSON needs no evaluator"
        raise ImportError(msg) from error
    ground_truth = coco_class(str(annotations))
    ground_truth.dataset.setdefault("info", {})
    if predictions:
        # loadRes adds IDs, areas and segmentation fields to the supplied objects.
        results = ground_truth.loadRes(copy.deepcopy(list(predictions)))
    else:
        # loadRes indexes its first result, so construct empty detections explicitly.
        results = coco_class()
        results.dataset = {
            "images": copy.deepcopy(ground_truth.dataset["images"]),
            "categories": copy.deepcopy(ground_truth.dataset["categories"]),
            "annotations": [],
        }
        results.createIndex()
    evaluator = evaluator_class(ground_truth, results, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        name: float(value) * 100 if value >= 0 else None
        for name, value in zip(METRIC_NAMES, evaluator.stats, strict=True)
    }
