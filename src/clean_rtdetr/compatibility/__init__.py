# Copyright 2026 Clean RT-DETR contributors
"""Strict official checkpoint compatibility for R50vd and individual decoder stages."""

from clean_rtdetr.compatibility.checkpoint import (
    CheckpointLoadReport,
    load_official_model_weights,
    load_official_stage_weights,
)

__all__ = ["CheckpointLoadReport", "load_official_model_weights", "load_official_stage_weights"]
