# Copyright 2026 Clean RT-DETR contributors
"""Parity tests for the narrow image inference boundaries."""

import torch
from PIL import Image

from clean_rtdetr.core.contracts import DetectionOutput
from clean_rtdetr.data import preprocess_images
from clean_rtdetr.postprocessing import postprocess
from tests.inference_oracle import postprocessor_oracle, preprocess_reference


def test_postprocess_matches_upstream_on_non_square_random_batch() -> None:
    """Compare labels, scores, and image-scale boxes with the upstream forward."""
    generator = torch.Generator().manual_seed(71)
    logits = torch.randn(2, 5, 3, generator=generator)
    boxes = torch.rand(2, 5, 4, generator=generator)
    sizes = torch.tensor([[31, 17], [7, 101]], dtype=torch.int64)
    top_k = 7

    expected = postprocessor_oracle(num_classes=3, num_top_queries=top_k).forward(
        {"pred_logits": logits, "pred_boxes": boxes}, sizes
    )
    actual = postprocess(DetectionOutput(logits, boxes), sizes, top_k=top_k)

    for batch_index, result in enumerate(expected):
        torch.testing.assert_close(actual.labels[batch_index], result["labels"])
        torch.testing.assert_close(actual.scores[batch_index], result["scores"])
        torch.testing.assert_close(actual.boxes_xyxy[batch_index], result["boxes"])


def test_postprocess_matches_upstream_for_tied_scores() -> None:
    """Keep upstream top-k tie behavior and flattened query/class ordering."""
    logits = torch.zeros(2, 4, 2)
    boxes = torch.tensor(
        [
            [[0.5, 0.5, 0.2, 0.4]] * 4,
            [[0.25, 0.75, 0.8, 0.2]] * 4,
        ],
        dtype=torch.float32,
    )
    sizes = torch.tensor([[19, 23], [101, 7]], dtype=torch.int64)
    expected = postprocessor_oracle(num_classes=2, num_top_queries=5).forward(
        {"pred_logits": logits, "pred_boxes": boxes}, sizes
    )
    actual = postprocess(DetectionOutput(logits, boxes), sizes, top_k=5)

    for batch_index, result in enumerate(expected):
        torch.testing.assert_close(actual.labels[batch_index], result["labels"])
        torch.testing.assert_close(actual.scores[batch_index], result["scores"])
        torch.testing.assert_close(actual.boxes_xyxy[batch_index], result["boxes"])


def test_preprocess_matches_documented_pil_validation_reference() -> None:
    """Validate RGB resize and byte-to-float behavior against a direct reference."""
    image = Image.new("RGBA", (9, 5))
    pixels = image.load()
    assert pixels is not None
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 31) % 256, (y * 47) % 256, (x + y) * 13 % 256, 127)

    actual = preprocess_images([image])
    expected = preprocess_reference(image)
    torch.testing.assert_close(actual.images[0], expected)
    torch.testing.assert_close(actual.original_sizes_wh[0], torch.tensor([9, 5]))
