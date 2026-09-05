# Copyright 2026 Clean RT-DETR contributors
"""Translate the official 80-class label space and evaluate COCO bbox results.

Category order follows the pinned RT-DETR COCO label mapping, documented in
docs/evaluation.md. Metric arithmetic is delegated to the official COCO API.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Int64
from torch import Tensor

from clean_rtdetr.postprocessing import Detections

COCO_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


@dataclass(frozen=True)
class CocoImage:
    """Describe one annotated source image without loading pixels."""

    image_id: int
    file_name: str
    width: int
    height: int


class CocoPrediction(TypedDict):
    """One COCO result: category ID, absolute xywh box, and probability score."""

    image_id: int
    category_id: int
    bbox: list[float]
    score: float


def load_coco_images(annotations: Path) -> list[CocoImage]:
    """Validate the official category space and return images in numeric ID order."""
    dataset = json.loads(annotations.read_text())
    if not isinstance(dataset, dict):
        msg = "COCO annotations must be a JSON object"
        raise ValueError(msg)  # noqa: TRY004 - invalid JSON contents
    categories = dataset.get("categories", [])
    if not isinstance(categories, list):
        msg = "COCO categories must be a list"
        raise ValueError(msg)  # noqa: TRY004 - invalid JSON contents
    ids = [category.get("id") for category in categories if isinstance(category, dict)]
    if (
        len(ids) != len(COCO_CATEGORY_IDS)
        or len(ids) != len(categories)
        or any(type(value) is not int for value in ids)
        or sorted(ids) != list(COCO_CATEGORY_IDS)
    ):
        msg = "Annotations must define exactly the official 80 COCO category IDs"
        raise ValueError(msg)
    images = dataset.get("images")
    if not isinstance(images, list) or not images:
        msg = "COCO annotations must contain a nonempty images list"
        raise ValueError(msg)
    result = []
    seen = set()
    for item in images:
        if (
            not isinstance(item, dict)
            or type(item.get("id")) is not int
            or item["id"] < 0
            or not isinstance(item.get("file_name"), str)
            or not item["file_name"]
            or any(type(item.get(key)) is not int or item[key] <= 0 for key in ("width", "height"))
        ):
            msg = "COCO images require nonnegative integer IDs, filenames, and positive dimensions"
            raise ValueError(msg)
        if item["id"] in seen:
            msg = "COCO image IDs must be unique"
            raise ValueError(msg)
        seen.add(item["id"])
        result.append(CocoImage(item["id"], item["file_name"], item["width"], item["height"]))
    return sorted(result, key=lambda image: image.image_id)


def coco_predictions(
    detections: Detections, image_ids: Int64[Tensor, " batch"]
) -> list[CocoPrediction]:
    """Map labels and convert xyxy to xywh without clipping, filtering, or NMS."""
    labels, scores, boxes = detections.labels, detections.scores, detections.boxes_xyxy
    if (
        image_ids.ndim != 1
        or image_ids.dtype != torch.int64
        or image_ids.numel() == 0
        or labels.ndim != 2  # noqa: PLR2004 - batch and detections
        or labels.shape[0] != image_ids.numel()
        or scores.shape != labels.shape
        or boxes.shape != (*labels.shape, 4)
    ):
        msg = "Expected image IDs [B], labels/scores [B,K], and boxes [B,K,4]"
        raise ValueError(msg)
    if (
        labels.dtype != torch.int64
        or not scores.is_floating_point()
        or not boxes.is_floating_point()
        or scores.dtype != boxes.dtype
        or not (labels.device == scores.device == boxes.device)
    ):
        msg = "Labels must be int64; scores and boxes must share floating dtype and device"
        raise ValueError(msg)
    if (
        not bool(torch.isfinite(scores).all() and torch.isfinite(boxes).all())
        or bool(((scores < 0) | (scores > 1)).any())
        or bool(((labels < 0) | (labels >= len(COCO_CATEGORY_IDS))).any())
        or bool((boxes[..., 2:] < boxes[..., :2]).any())
        or bool((image_ids < 0).any())
        or image_ids.unique().numel() != image_ids.numel()
    ):
        msg = "Invalid COCO IDs, labels, probabilities, or finite nonnegative box extents"
        raise ValueError(msg)
    xywh = torch.cat((boxes[..., :2], boxes[..., 2:] - boxes[..., :2]), dim=-1)
    if not bool(torch.isfinite(xywh).all()):
        msg = "COCO xywh conversion must produce finite box extents"
        raise ValueError(msg)
    result: list[CocoPrediction] = []
    for image_id, image_labels, image_scores, image_boxes in zip(
        image_ids.cpu().tolist(),
        labels.cpu().tolist(),
        scores.detach().cpu().tolist(),
        xywh.detach().cpu().tolist(),
        strict=True,
    ):
        result.extend(
            CocoPrediction(
                image_id=image_id, category_id=COCO_CATEGORY_IDS[label], bbox=box, score=score
            )
            for label, score, box in zip(image_labels, image_scores, image_boxes, strict=True)
        )
    return result


def _validate_predictions(predictions: Sequence[CocoPrediction], image_ids: set[int]) -> None:
    for prediction in predictions:
        if not isinstance(prediction, dict):
            msg = "Each COCO prediction must be an object"
            raise ValueError(msg)  # noqa: TRY004 - invalid result JSON contents
        box = prediction.get("bbox")
        score = prediction.get("score")
        if (
            type(prediction.get("image_id")) is not int
            or prediction["image_id"] not in image_ids
            or type(prediction.get("category_id")) is not int
            or prediction["category_id"] not in COCO_CATEGORY_IDS
            or not isinstance(box, list)
            or len(box) != 4  # noqa: PLR2004 - xywh
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
            or box[2] < 0
            or box[3] < 0
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            msg = "Predictions require selected image IDs, COCO categories, finite xywh and scores"
            raise ValueError(msg)


def evaluate_coco(
    annotations: Path,
    predictions: Sequence[CocoPrediction],
    *,
    image_ids: Sequence[int] | None = None,
) -> dict[str, float | None]:
    """Return official bbox AP/AR in 0-100 units; undefined area metrics are None.

    All annotated images are evaluated by default, including images with no
    detections. Explicit subsets must contain unique, known image IDs. COCO's
    standard IoU/recall thresholds and maxDets=[1,10,100] remain unchanged.
    """
    available = {image.image_id for image in load_coco_images(annotations)}
    selected = sorted(available) if image_ids is None else list(image_ids)
    if (
        not selected
        or any(type(value) is not int for value in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= available
    ):
        msg = "Evaluation requires nonempty, unique image IDs present in the annotations"
        raise ValueError(msg)
    _validate_predictions(predictions, set(selected))
    from clean_rtdetr.evaluation._pycocotools import evaluate_bbox  # noqa: PLC0415

    return evaluate_bbox(annotations, predictions, sorted(selected))
