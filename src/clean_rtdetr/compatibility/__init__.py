# Copyright 2026 Clean RT-DETR contributors
"""Strict official checkpoint compatibility for the implemented decoder stages."""

from clean_rtdetr.compatibility.checkpoint import (
    CheckpointLoadReport,
    load_official_stage_weights,
)

__all__ = ["CheckpointLoadReport", "load_official_stage_weights"]
