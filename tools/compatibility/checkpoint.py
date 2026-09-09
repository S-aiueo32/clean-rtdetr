# Copyright 2026 Clean RT-DETR contributors
"""Checkout-only compatibility loader for official stage weights."""

import hashlib
import io
from pathlib import Path
from typing import Literal

import torch

from clean_rtdetr.cli._official_checkpoint import (
    CheckpointLoadReport,
    _convert_modules,
    _source_key,
    _state_dict,
    _validate_modules,
)
from clean_rtdetr.models.decoder import Decoder
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.selector import QuerySelector


def _excluded_reason(key: str) -> str | None:
    """Describe explicitly unsupported full-model components."""
    if key.startswith("backbone."):
        return "Backbone is not requested by the stage loader"
    if key.startswith("encoder."):
        return "HybridEncoder is not requested by the stage loader"
    if key == "decoder.denoising_class_embed.weight":
        return "Training denoising embedding is not requested by the stage loader"
    return None


def load_official_stage_weights(  # noqa: PLR0913
    path: str | Path,
    *,
    projector: MemoryProjector,
    selector: QuerySelector,
    decoder: Decoder,
    source: Literal["model", "ema"],
    expected_sha256: str,
) -> CheckpointLoadReport:
    """Load an explicit official weight source after validating all three stages.

    Only the pinned RT-DETR parameter layout and default component structures are
    supported. Tensor shapes and dtypes must match exactly. Missing BatchNorm
    counters in Paddle-converted weights are initialized to zero and reported;
    every other target tensor is required. Optimizer/resume state is not loaded.
    """
    _validate_modules(projector, selector, decoder)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        msg = f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        raise ValueError(msg)
    state = _state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), source)
    modules = {"projector": projector, "selector": selector, "decoder": decoder}
    converted, mapping, synthesized = _convert_modules(state, modules, _source_key)
    excluded: dict[str, str] = {}
    for key in state.keys() - mapping.keys():
        reason = _excluded_reason(key)
        if reason is None:
            msg = f"Unexpected checkpoint key: {key}"
            raise ValueError(msg)
        excluded[key] = reason
    # Every stage has been checked before the first parameter or buffer is changed.
    for stage, module in modules.items():
        module.load_state_dict(converted[stage], strict=True)
    return CheckpointLoadReport(digest, source, mapping, excluded, tuple(synthesized))
