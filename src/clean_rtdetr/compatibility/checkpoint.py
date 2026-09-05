# Copyright 2026 Clean RT-DETR contributors
"""Validate and split official RT-DETR weights before changing stage modules."""

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from clean_rtdetr.models.decoder import Decoder, DecoderLayer
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.selector import BoxHead, QuerySelector


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Record the selected artifact and every mapped, excluded, or supplied key."""

    sha256: str
    source: Literal["model", "ema"]
    key_mapping: dict[str, str]
    excluded_keys: dict[str, str]
    synthesized_keys: tuple[str, ...]


def _validate_modules(
    projector: MemoryProjector, selector: QuerySelector, decoder: Decoder
) -> None:
    """Limit conversion to compatible default stage structures."""
    if (
        type(projector) is not MemoryProjector
        or type(selector) is not QuerySelector
        or type(decoder) is not Decoder
        or type(selector.enc_score_head) is not nn.Linear
        or type(selector.enc_bbox_head) is not BoxHead
        or any(type(layer) is not DecoderLayer for layer in decoder.layers)
    ):
        msg = "Official conversion requires default stage modules and prediction heads"
        raise ValueError(msg)
    if (
        projector.hidden_dim != selector.config.hidden_dim
        or projector.hidden_dim != decoder.config.hidden_dim
        or selector.config.num_classes != decoder.config.num_classes
        or len(projector.specs) != decoder.config.num_levels
    ):
        msg = "Projector, selector, and decoder configurations are incompatible"
        raise ValueError(msg)


def _state_dict(payload: object, source: Literal["model", "ema"]) -> Mapping[str, Tensor]:
    """Select an explicit official envelope without trying alternate sources."""
    if source not in {"model", "ema"}:
        msg = "Checkpoint source must be model or ema"
        raise ValueError(msg)
    if not isinstance(payload, dict) or source not in payload:
        msg = f"Checkpoint has no {source!r} state"
        raise ValueError(msg)
    state = payload[source]
    if source == "ema":
        if not isinstance(state, dict) or "module" not in state:
            msg = "Checkpoint EMA must contain a module state dict"
            raise ValueError(msg)
        state = state["module"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        msg = "Selected checkpoint state must map string keys to tensors"
        raise ValueError(msg)
    return state


def _source_key(stage: str, target: str) -> str:
    """Translate one known target namespace to the upstream model namespace."""
    if stage == "decoder" and target.startswith("layers."):
        return f"decoder.decoder.{target}"
    return f"decoder.{target}"


def _excluded_reason(key: str) -> str | None:
    """Describe explicitly unsupported full-model components."""
    if key.startswith("backbone."):
        return "Backbone is not implemented"
    if key.startswith("encoder."):
        return "HybridEncoder is not implemented"
    if key == "decoder.denoising_class_embed.weight":
        return "Training denoising embedding is not implemented"
    return None


def _validate_tensor(key: str, value: Tensor, expected: Tensor) -> None:
    """Reject incompatible weights before any destination module is mutated."""
    if value.layout != torch.strided or value.is_meta or expected.is_meta:
        msg = f"Checkpoint loading requires materialized strided tensors: {key}"
        raise ValueError(msg)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        msg = (
            f"Checkpoint shape/dtype mismatch for {key}: "
            f"expected {tuple(expected.shape)} {expected.dtype}, "
            f"got {tuple(value.shape)} {value.dtype}"
        )
        raise ValueError(msg)


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
    converted: dict[str, dict[str, Tensor]] = {}
    mapping: dict[str, str] = {}
    synthesized: list[str] = []
    for stage, module in modules.items():
        converted[stage] = {}
        for target, expected in module.state_dict().items():
            key = _source_key(stage, target)
            destination = f"{stage}.{target}"
            if key in mapping:
                msg = f"Duplicate checkpoint key mapping: {key}"
                raise ValueError(msg)
            if key not in state:
                if stage == "projector" and target.endswith(".norm.num_batches_tracked"):
                    converted[stage][target] = torch.zeros_like(expected)
                    synthesized.append(destination)
                    continue
                msg = f"Missing checkpoint key: {key}"
                raise ValueError(msg)
            value = state[key]
            _validate_tensor(key, value, expected)
            converted[stage][target] = value
            mapping[key] = destination
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
