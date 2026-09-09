# Copyright 2026 Clean RT-DETR contributors
"""COCO bounding-box evaluation with optional pycocotools dependencies."""

from clean_rtdetr.evaluation.coco import (
    COCO_CATEGORY_IDS,
    CocoImage,
    CocoPrediction,
    coco_predictions,
    evaluate_coco,
    load_coco_images,
)

__all__ = [
    "COCO_CATEGORY_IDS",
    "CocoImage",
    "CocoPrediction",
    "coco_predictions",
    "evaluate_coco",
    "load_coco_images",
]
