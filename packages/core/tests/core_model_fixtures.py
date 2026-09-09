# Copyright 2026 Clean RT-DETR contributors
"""Model factories for core package tests."""

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
    rtdetr_r50vd,
)
from clean_rtdetr.models.rtdetr import BaselineRTDETR


def small_model() -> BaselineRTDETR:
    """Build a compact model retaining all production boundaries."""
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
    return RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)


def baseline_model() -> BaselineRTDETR:
    """Build the pinned R50vd model recipe."""
    return rtdetr_r50vd()
