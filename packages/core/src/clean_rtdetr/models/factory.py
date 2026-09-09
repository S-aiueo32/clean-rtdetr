# Copyright 2026 Clean RT-DETR contributors
"""Pinned baseline model factories."""

from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.rtdetr import RTDETR, BaselineRTDETR
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig


def rtdetr_r50vd() -> BaselineRTDETR:
    """Build the pinned 80-class R50vd inference baseline."""
    backbone = PResNet(freeze_at=0, freeze_norm=True)
    encoder = HybridEncoder(backbone.specs, HybridEncoderConfig(hidden_dim=256))
    projector = MemoryProjector(encoder.output_specs, hidden_dim=256)
    selector = QuerySelector(
        QuerySelectorConfig(
            hidden_dim=256,
            num_classes=80,
            num_queries=300,
            learned_content=False,
        )
    )
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=256,
            num_classes=80,
            num_layers=6,
            num_heads=8,
            num_levels=3,
        )
    )
    return RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)
