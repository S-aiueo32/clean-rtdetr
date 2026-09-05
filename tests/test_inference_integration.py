# Copyright 2026 Clean RT-DETR contributors
"""Exercise real preprocessing, model, and postprocessing boundaries together."""

import torch
from PIL import Image

from clean_rtdetr import DecoderPlan
from clean_rtdetr.data import preprocess_images
from clean_rtdetr.postprocessing import postprocess
from tests.test_model import small_model


def test_image_to_detections() -> None:
    """A small injected model consumes the actual fixed-size preprocessing output."""
    batch = preprocess_images([Image.new("RGB", (91, 47), (32, 127, 240))])
    model = small_model().eval()
    with torch.inference_mode():
        output = model(batch.images, plan=DecoderPlan(collect_intermediate=False))
        detections = postprocess(output.final, batch.original_sizes_wh, top_k=3)
    assert detections.labels.shape == (1, 3)
    assert detections.boxes_xyxy.shape == (1, 3, 4)
    assert torch.isfinite(detections.scores).all()
    assert torch.isfinite(detections.boxes_xyxy).all()
    assert batch.original_sizes_wh.tolist() == [[91, 47]]
