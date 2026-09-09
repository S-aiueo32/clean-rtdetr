# Copyright 2026 Clean RT-DETR contributors
"""Load official detector and denoising weights without coupling core to training."""

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Shaped
from torch import Tensor

from clean_rtdetr.compatibility import CheckpointLoadReport, load_official_model_weights
from clean_rtdetr.models.rtdetr import BaselineRTDETR
from clean_rtdetr_training.denoising import DenoisingQueryGenerator, DenoisingTrainingModel

_DN_SOURCE_KEY = "decoder.denoising_class_embed.weight"
_DN_TARGET_KEY = "class_embed.weight"


def _selected_state(
    data: bytes, source: Literal["model", "ema"]
) -> Mapping[str, Shaped[Tensor, "..."]]:
    """Select and validate the requested official state envelope."""
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
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


def _validate_denoising_target(
    model: DenoisingTrainingModel[BaselineRTDETR, DenoisingQueryGenerator],
    state: Mapping[str, Shaped[Tensor, "..."]],
) -> Shaped[Tensor, "classes hidden"]:
    """Validate the default DN embedding before any destination is mutated."""
    if type(model.denoising) is not DenoisingQueryGenerator:
        msg = "Official conversion requires the default denoising query generator"
        raise ValueError(msg)
    target_state = model.denoising.state_dict()
    if tuple(target_state) != (_DN_TARGET_KEY,):
        msg = "Official conversion requires only the default denoising embedding state"
        raise ValueError(msg)
    if _DN_SOURCE_KEY not in state:
        msg = f"Missing checkpoint key: {_DN_SOURCE_KEY}"
        raise ValueError(msg)
    value = state[_DN_SOURCE_KEY]
    expected = target_state[_DN_TARGET_KEY]
    if value.layout != torch.strided or value.is_meta or expected.is_meta:
        msg = f"Checkpoint loading requires materialized strided tensors: {_DN_SOURCE_KEY}"
        raise ValueError(msg)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        msg = (
            f"Checkpoint shape/dtype mismatch for {_DN_SOURCE_KEY}: "
            f"expected {tuple(expected.shape)} {expected.dtype}, "
            f"got {tuple(value.shape)} {value.dtype}"
        )
        raise ValueError(msg)
    return value


def load_official_training_weights(
    path: str | Path,
    *,
    model: DenoisingTrainingModel[BaselineRTDETR, DenoisingQueryGenerator],
    source: Literal["model", "ema"],
    expected_sha256: str,
) -> CheckpointLoadReport:
    """Atomically validate and load official detector and DN embedding weights."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        msg = f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        raise ValueError(msg)
    state = _selected_state(data, source)
    denoising_weight = _validate_denoising_target(model, state)

    report = load_official_model_weights(
        path,
        model=model.model,
        source=source,
        expected_sha256=expected_sha256,
    )
    model.denoising.load_state_dict({_DN_TARGET_KEY: denoising_weight}, strict=True)
    mapping = {source: f"model.{target}" for source, target in report.key_mapping.items()}
    mapping[_DN_SOURCE_KEY] = f"denoising.{_DN_TARGET_KEY}"
    excluded = dict(report.excluded_keys)
    excluded.pop(_DN_SOURCE_KEY, None)
    return CheckpointLoadReport(
        report.sha256,
        report.source,
        mapping,
        excluded,
        tuple(f"model.{key}" for key in report.synthesized_keys),
    )
