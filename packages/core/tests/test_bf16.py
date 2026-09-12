# Copyright 2026 Clean RT-DETR contributors
"""Mixed precision preserves public prediction and geometry contracts."""

import pytest
import torch
from core_model_fixtures import small_model

from clean_rtdetr import QuerySelector, QuerySelectorConfig
from clean_rtdetr.inference import postprocess


@pytest.mark.parametrize("learned_content", [False, True])
def test_bf16_autocast_inference(*, learned_content: bool) -> None:
    """Use bf16 feature projections and FP32 predictions through postprocessing."""
    torch.manual_seed(17)
    model = small_model().eval()
    model.selector = QuerySelector(
        QuerySelectorConfig(
            hidden_dim=8, num_classes=3, num_queries=2, learned_content=learned_content
        )
    )
    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = model(images)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            memory = model.projector(model.encoder(model.backbone(images)).features)
            assert memory.memory.dtype == torch.bfloat16
            actual = model(images)
        actual.decoder.validate()
        assert actual.final.logits.dtype == actual.final.boxes.dtype == torch.float32
        assert actual.encoder_aux.logits.dtype == torch.float32
        assert torch.isfinite(actual.final.logits).all()
        torch.testing.assert_close(actual.final.logits, expected.final.logits, atol=0.1, rtol=0.05)
        # Quantized ranking can select different anchors for near-tied scores.
        assert torch.isfinite(actual.final.boxes).all()
        assert ((actual.final.boxes >= 0) & (actual.final.boxes <= 1)).all()
        detections = postprocess(actual.final, torch.tensor([[64, 64]]), top_k=2)
        assert torch.isfinite(detections.boxes_xyxy).all()
