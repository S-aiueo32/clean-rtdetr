# Copyright 2026 Clean RT-DETR contributors
"""Checkout-only compatibility for official R50vd checkpoint stages."""

from clean_rtdetr.cli._official_checkpoint import (
    CheckpointLoadReport,
    load_official_model_weights,
)
from tools.compatibility.checkpoint import load_official_stage_weights

__all__ = ["CheckpointLoadReport", "load_official_model_weights", "load_official_stage_weights"]
