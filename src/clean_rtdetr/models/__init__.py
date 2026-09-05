# Copyright 2026 Clean RT-DETR contributors
"""Concrete model components with constructor injection."""

from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.interfaces import (
    Assembler,
    Backbone,
    Decoder,
    Detector,
    Encoder,
    Projector,
    Selector,
)
from clean_rtdetr.models.rtdetr import RTDETR

__all__ = [
    "RTDETR",
    "Assembler",
    "Backbone",
    "Decoder",
    "Detector",
    "Encoder",
    "HybridEncoder",
    "HybridEncoderConfig",
    "PResNet",
    "Projector",
    "Selector",
]
