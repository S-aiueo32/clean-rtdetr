# Copyright 2026 Clean RT-DETR contributors
"""Explicit RT-DETR components with a separate image inference API."""

from clean_rtdetr.contracts import (
    DecoderInput,
    DecoderOutput,
    DenoisingQueries,
    DetectionOutput,
    FeatureSpec,
    HybridEncoderOutput,
    ModelOutput,
    MultiScaleFeatures,
    QuerySet,
)
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig, DecoderPlan
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
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.rtdetr import RTDETR
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig

__all__ = [
    "RTDETR",
    "Decoder",
    "DecoderConfig",
    "DecoderInput",
    "DecoderOutput",
    "DecoderPlan",
    "DenoisingQueries",
    "DetectionOutput",
    "FeatureSpec",
    "HybridEncoder",
    "HybridEncoderConfig",
    "HybridEncoderOutput",
    "MemoryProjector",
    "ModelFactory",
    "ModelOutput",
    "ModelPreset",
    "MultiScaleFeatures",
    "PResNet",
    "QueryAssembler",
    "QuerySelector",
    "QuerySelectorConfig",
    "QuerySet",
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
