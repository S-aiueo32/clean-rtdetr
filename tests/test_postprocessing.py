# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for decoder-to-detection postprocessing."""

from typing import cast

import pytest
import torch
from jaxtyping import TypeCheckError

from clean_rtdetr.core.contracts import DetectionOutput
from clean_rtdetr.postprocessing import Detections, postprocess


def test_multilabel_scaling_and_out_of_image_boxes() -> None:
    """Flattened query/class ranking allows multiple labels per query."""
    logits = torch.tensor([[[4.0, 3.0], [-4.0, 2.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.2], [0.2, 0.8, 1.4, 0.5]]])
    result = postprocess(DetectionOutput(logits, boxes), torch.tensor([[100, 200]]), top_k=3)
    assert isinstance(result, Detections)
    torch.testing.assert_close(result.labels, torch.tensor([[0, 1, 1]]))
    torch.testing.assert_close(result.scores, logits.sigmoid().flatten().topk(3).values[None])
    expected = torch.tensor(
        [
            [
                [30.0, 80.0, 70.0, 120.0],
                [30.0, 80.0, 70.0, 120.0],
                [-50.0, 110.0, 90.0, 210.0],
            ]
        ]
    )
    torch.testing.assert_close(result.boxes_xyxy, expected)


def test_ties_are_sorted_and_limit_is_strict() -> None:
    """Ties still produce the requested number of sorted top-k scores."""
    output = DetectionOutput(torch.zeros(1, 2, 2), torch.ones(1, 2, 4))
    result = postprocess(output, torch.tensor([[10, 20]]), top_k=2)
    torch.testing.assert_close(result.scores, torch.full((1, 2), 0.5))
    with pytest.raises(ValueError, match="top_k"):
        postprocess(output, torch.tensor([[10, 20]]), top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        postprocess(output, torch.tensor([[10, 20]]), top_k=5)


def test_limits_and_inputs_are_preserved() -> None:
    """Accept both top-k limits without mutating prediction or size tensors."""
    logits = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.4, 0.6, 0.3, 0.4]]])
    sizes = torch.tensor([[80, 40]], dtype=torch.int64)
    logits_before, boxes_before, sizes_before = logits.clone(), boxes.clone(), sizes.clone()
    one = postprocess(DetectionOutput(logits, boxes), sizes, top_k=1)
    all_detections = postprocess(DetectionOutput(logits, boxes), sizes, top_k=4)
    assert one.labels.shape == (1, 1)
    assert all_detections.labels.shape == (1, 4)
    assert torch.equal(logits, logits_before)
    assert torch.equal(boxes, boxes_before)
    assert torch.equal(sizes, sizes_before)


def test_nan_logits_are_rejected() -> None:
    """Reject non-finite logits before ranking."""
    logits = torch.tensor([[[float("nan")]]])
    with pytest.raises(ValueError, match="finite"):
        postprocess(
            DetectionOutput(logits, torch.ones(1, 1, 4)),
            torch.tensor([[10, 20]], dtype=torch.int64),
        )


@pytest.mark.parametrize("kind", ["shape", "dtype", "device"])
def test_prediction_contract_mismatches_are_rejected(kind: str) -> None:
    """Reject mismatched prediction fields even when a malformed dataclass is supplied."""
    logits = torch.ones(1, 1, 1)
    boxes = torch.ones(1, 1, 4)
    malformed = DetectionOutput(logits, boxes)
    if kind == "shape":
        object.__setattr__(malformed, "boxes", torch.ones(1, 2, 4))
    elif kind == "dtype":
        object.__setattr__(malformed, "boxes", boxes.to(torch.float64))
    else:
        object.__setattr__(malformed, "boxes", boxes.to("meta"))
    with pytest.raises(ValueError, match=r"share|same device"):
        postprocess(malformed, torch.tensor([[10, 20]], dtype=torch.int64))


@pytest.mark.parametrize(
    ("predictions", "sizes", "top_k"),
    [
        (
            DetectionOutput(torch.empty(0, 1, 1), torch.empty(0, 1, 4)),
            torch.ones(0, 2, dtype=torch.int64),
            1,
        ),
        (
            DetectionOutput(torch.ones(1, 1, 1), torch.ones(1, 1, 4)),
            torch.ones(1, 3, dtype=torch.int64),
            1,
        ),
        (DetectionOutput(torch.ones(1, 1, 1), torch.ones(1, 1, 4)), torch.ones(1, 2), 1),
        (
            DetectionOutput(torch.ones(1, 1, 1), torch.ones(1, 1, 4)),
            torch.zeros(1, 2, dtype=torch.int64),
            1,
        ),
        (
            DetectionOutput(torch.ones(1, 1, 1), torch.full((1, 1, 4), float("inf"))),
            torch.ones(1, 2, dtype=torch.int64),
            1,
        ),
    ],
)
def test_validation_rejects_invalid_inputs(
    predictions: DetectionOutput, sizes: torch.Tensor, top_k: int
) -> None:
    """Reject invalid sizes and non-finite prediction tensors."""
    with pytest.raises(ValueError, match=r"must|shape"):
        postprocess(predictions, sizes, top_k=top_k)


@pytest.mark.parametrize("top_k", [True, 1.0, torch.tensor(1)])
def test_top_k_must_be_strict_integer(top_k: object) -> None:
    """Reject bools and integer-like values that are not actual Python ints."""
    output = DetectionOutput(torch.ones(1, 1, 1), torch.ones(1, 1, 4))
    exception = ValueError if isinstance(top_k, bool) else TypeCheckError
    with pytest.raises(exception):
        postprocess(
            output,
            torch.ones(1, 2, dtype=torch.int64),
            top_k=cast("int", top_k),
        )
