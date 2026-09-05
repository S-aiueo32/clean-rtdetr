# Copyright 2026 Clean RT-DETR contributors
"""Tests for COCO metadata, conversion, and optional official evaluation."""

import ast
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from clean_rtdetr.evaluation import (
    COCO_CATEGORY_IDS,
    coco_predictions,
    evaluate_coco,
    load_coco_images,
)
from clean_rtdetr.postprocessing import Detections

AP_PERFECT = 100.0


def annotation_data() -> dict[str, object]:
    """Return a tiny two-image dataset with the complete official category list."""
    return {
        "images": [
            {"id": 2, "file_name": "two.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": value, "name": str(value)} for value in COCO_CATEGORY_IDS],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 13,
                "bbox": [30, 40, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }


def write_annotations(tmp_path: Path, data: dict[str, object] | None = None) -> Path:
    """Write fixture annotations and return their path."""
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotation_data() if data is None else data))
    return path


def detections() -> Detections:
    """Return perfect detections for both fixture images."""
    return Detections(
        labels=torch.tensor([[0], [11]], dtype=torch.int64),
        scores=torch.ones((2, 1)),
        boxes_xyxy=torch.tensor([[[10, 20, 30, 40]], [[30, 40, 50, 60]]], dtype=torch.float32),
    )


def test_load_images_sorts_and_validates_categories(tmp_path: Path) -> None:
    """Metadata is returned in stable numeric order."""
    images = load_coco_images(write_annotations(tmp_path))
    assert [(item.image_id, item.file_name) for item in images] == [(1, "one.jpg"), (2, "two.jpg")]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["categories"].pop(),
        lambda data: data["images"].append(data["images"][0]),
        lambda data: data["images"][0].update(id=-1),
        lambda data: data["images"][0].update(width=0),
    ],
)
def test_load_images_rejects_invalid_metadata(
    tmp_path: Path, mutation: Callable[[dict[str, object]], object]
) -> None:
    """Reject malformed category lists, duplicate IDs, and invalid dimensions."""
    data = annotation_data()
    mutation(data)
    with pytest.raises(ValueError, match="COCO"):
        load_coco_images(write_annotations(tmp_path, data))


def test_coco_predictions_maps_labels_and_does_not_clip() -> None:
    """Map zero-based labels and convert xyxy to xywh without clipping."""
    result = coco_predictions(detections(), torch.tensor([1, 2], dtype=torch.int64))
    assert [item["category_id"] for item in result] == [1, 13]
    assert result[0]["bbox"] == [10.0, 20.0, 20.0, 20.0]
    negative = Detections(
        labels=torch.tensor([[79]], dtype=torch.int64),
        scores=torch.ones((1, 1)),
        boxes_xyxy=torch.tensor([[[-5, -4, 6, 7]]], dtype=torch.float32),
    )
    assert coco_predictions(negative, torch.tensor([1]))[0] == {
        "image_id": 1,
        "category_id": 90,
        "bbox": [-5.0, -4.0, 11.0, 11.0],
        "score": 1.0,
    }


def test_coco_predictions_allows_empty_detections() -> None:
    """An empty per-image matrix converts to an empty result list."""
    empty = Detections(
        torch.empty((2, 0), dtype=torch.int64), torch.empty((2, 0)), torch.empty((2, 0, 4))
    )
    assert coco_predictions(empty, torch.tensor([1, 2], dtype=torch.int64)) == []


@pytest.mark.parametrize(
    "bad",
    [
        Detections(torch.tensor([[80]]), torch.ones((1, 1)), torch.zeros((1, 1, 4))),
        Detections(torch.tensor([[0]]), torch.tensor([[float("nan")]]), torch.zeros((1, 1, 4))),
        Detections(torch.tensor([[0]]), torch.ones((1, 1)), torch.tensor([[[2, 2, 1, 3.0]]])),
    ],
)
def test_coco_predictions_rejects_invalid_values(bad: Detections) -> None:
    """Reject invalid labels, scores, and extents."""
    with pytest.raises(ValueError, match=r"Invalid|Labels"):
        coco_predictions(bad, torch.tensor([1], dtype=torch.int64))


def test_public_conversion_has_no_pycocotools_import() -> None:
    """Keep the optional dependency isolated in the private backend module."""
    root = Path(__file__).parents[1] / "src/clean_rtdetr"
    for source in root.rglob("*.py"):
        if source.name != "_pycocotools.py":
            tree = ast.parse(source.read_text())
            imports = [
                node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            for node in imports:
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                assert all(not name.startswith("pycocotools") for name in names)
            assert not any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("pycocotools.")
                for node in ast.walk(tree)
            )


def test_conversion_works_when_pycocotools_is_blocked(tmp_path: Path) -> None:
    """Metadata and conversion do not require importing the optional evaluator."""
    annotations = write_annotations(tmp_path)
    script = """
import importlib.abc
import pathlib
import sys

class BlockEvaluator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('pycocotools'):
            raise AssertionError('pycocotools imported')

sys.meta_path.insert(0, BlockEvaluator())
import torch
from clean_rtdetr.evaluation import load_coco_images, coco_predictions
from clean_rtdetr.postprocessing import Detections
assert [x.image_id for x in load_coco_images(pathlib.Path(sys.argv[1]))] == [1, 2]
x = Detections(torch.tensor([[0], [11]]), torch.ones((2, 1)), torch.zeros((2, 1, 4)))
assert len(coco_predictions(x, torch.tensor([1, 2]))) == 2
"""
    subprocess.run(  # noqa: S603 - executable and script are test-local
        [sys.executable, "-c", script, str(annotations)], check=True
    )


@pytest.mark.parametrize(("predictions", "expected"), [(None, AP_PERFECT), ([], 0.0)])
def test_evaluate_perfect_and_empty(tmp_path: Path, predictions: object, expected: float) -> None:
    """Official evaluation reports AP100 for perfect results and AP0 for none."""
    pytest.importorskip("pycocotools")
    actual = coco_predictions(detections(), torch.tensor([1, 2])) if predictions is None else []
    metrics = evaluate_coco(write_annotations(tmp_path), actual)
    assert metrics["AP"] == pytest.approx(expected)


def test_evaluate_subset_and_undefined_area_metrics(tmp_path: Path) -> None:
    """Subset evaluation excludes missing images and exposes undefined area buckets."""
    pytest.importorskip("pycocotools")
    annotations = write_annotations(tmp_path)
    predictions = coco_predictions(detections(), torch.tensor([1, 2]))
    subset = evaluate_coco(annotations, predictions[:1], image_ids=[1])
    full = evaluate_coco(annotations, predictions[:1])
    assert subset["AP"] == pytest.approx(AP_PERFECT)
    assert full["AP"] is not None
    assert subset["AP"] is not None
    assert full["AP"] < subset["AP"]
    assert subset["AP_small"] == pytest.approx(AP_PERFECT)
    assert subset["AP_medium"] is None
    assert subset["AP_large"] is None


def test_evaluate_does_not_mutate_predictions(tmp_path: Path) -> None:
    """Evaluation deep-copies result records before the official API consumes them."""
    pytest.importorskip("pycocotools")
    predictions = coco_predictions(detections(), torch.tensor([1, 2]))
    before = json.loads(json.dumps(predictions))
    evaluate_coco(write_annotations(tmp_path), predictions)
    assert predictions == before


def test_evaluate_ignores_high_scoring_crowd_match(tmp_path: Path) -> None:
    """A crowd-only match must not become a high-ranking false positive."""
    pytest.importorskip("pycocotools")
    data = annotation_data()
    cast("list[dict[str, object]]", data["annotations"]).append(
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "bbox": [60, 60, 20, 20],
            "area": 400,
            "iscrowd": 1,
        }
    )
    predictions = coco_predictions(detections(), torch.tensor([1, 2]))
    for prediction in predictions:
        prediction["score"] = 0.9
    predictions.append(
        {"image_id": 1, "category_id": 1, "bbox": [60.0, 60.0, 20.0, 20.0], "score": 1.0}
    )
    result = evaluate_coco(write_annotations(tmp_path, data), predictions)
    assert result["AP"] == pytest.approx(AP_PERFECT)


@pytest.mark.parametrize("image_ids", [[], [1, 1], [3]])
def test_evaluate_rejects_invalid_subsets(tmp_path: Path, image_ids: list[int]) -> None:
    """Do not silently evaluate an empty, duplicate, or unknown image subset."""
    with pytest.raises(ValueError, match="image IDs"):
        evaluate_coco(write_annotations(tmp_path), [], image_ids=image_ids)


def test_evaluate_rejects_predictions_outside_subset(tmp_path: Path) -> None:
    """Selected images must account for every supplied prediction."""
    predictions = coco_predictions(detections(), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="selected image IDs"):
        evaluate_coco(write_annotations(tmp_path), predictions, image_ids=[1])
