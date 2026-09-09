# Copyright 2026 Clean RT-DETR contributors
"""Concrete model components with constructor injection."""

from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.factory import (
    ModelFactory,
    ModelPreset,
    create_model,
    get_model_preset,
    list_models,
    rtdetr_r18vd,
    rtdetr_r34vd,
    rtdetr_r50vd,
    rtdetr_r50vd_m,
    rtdetr_r101vd,
    rtdetrv2_r18vd,
    rtdetrv2_r34vd,
    rtdetrv2_r50vd,
    rtdetrv2_r50vd_m,
    rtdetrv2_r101vd,
)
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
    "ModelFactory",
    "ModelPreset",
    "PResNet",
    "Projector",
    "Selector",
    "create_model",
    "get_model_preset",
    "list_models",
    "rtdetr_r18vd",
    "rtdetr_r34vd",
    "rtdetr_r50vd",
    "rtdetr_r50vd_m",
    "rtdetr_r101vd",
    "rtdetrv2_r18vd",
    "rtdetrv2_r34vd",
    "rtdetrv2_r50vd",
    "rtdetrv2_r50vd_m",
    "rtdetrv2_r101vd",
]
