# Copyright 2026 Clean RT-DETR contributors
"""Offline image-to-prediction parity against independent upstream stages."""

import torch

from clean_rtdetr import (
    RTDETR,
    Decoder,
    DecoderConfig,
    HybridEncoder,
    HybridEncoderConfig,
    MemoryProjector,
    PResNet,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)
from tests.official_oracle import (
    backbone_oracle,
    full_transformer_oracle,
    hybrid_encoder_oracle,
)


@torch.inference_mode()
def test_image_predictions_match_pinned_upstream_without_checkpoint() -> None:
    """Exercise every boundary on rectangular images with upstream random weights."""
    torch.manual_seed(41)
    upstream_backbone = backbone_oracle(
        depth=50,
        variant="d",
        return_idx=[1, 2, 3],
        pretrained=False,
        freeze_at=0,
        freeze_norm=True,
    ).eval()
    upstream_encoder = hybrid_encoder_oracle(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
    ).eval()
    upstream_decoder = full_transformer_oracle(
        num_classes=3,
        hidden_dim=8,
        num_queries=2,
        feat_channels=[8, 8, 8],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=1,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        num_denoising=0,
    ).eval()
    backbone = PResNet()
    encoder = HybridEncoder(
        backbone.specs,
        HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
    )
    projector = MemoryProjector(encoder.output_specs, hidden_dim=8)
    selector = QuerySelector(QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=2))
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=8,
            num_classes=3,
            num_layers=2,
            num_heads=2,
            num_levels=3,
            num_points=1,
            feedforward_dim=16,
        )
    )
    backbone.load_state_dict(upstream_backbone.state_dict(), strict=True)
    encoder.load_state_dict(upstream_encoder.state_dict(), strict=True)
    upstream_state = upstream_decoder.state_dict()
    for stage in (projector, selector, decoder):
        converted = {
            key: upstream_state[f"decoder.{key}" if key.startswith("layers.") else key]
            for key in stage.state_dict()
        }
        stage.load_state_dict(converted, strict=True)
    model = RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder).eval()
    images = torch.randn(2, 3, 64, 96)
    actual = model(images)
    expected = upstream_decoder(upstream_encoder(upstream_backbone(images)))
    torch.testing.assert_close(actual.final.logits, expected["pred_logits"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual.final.boxes, expected["pred_boxes"], rtol=1e-4, atol=1e-5)
