# Copyright 2026 Clean RT-DETR contributors
"""Load native tensor state dictionaries without checkpoint format conversion."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Shaped
from torch import Tensor, nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Record the native checkpoint digest and its loaded tensor keys."""

    sha256: str
    keys: tuple[str, ...]


def _validate_tensor(
    key: str, value: Shaped[Tensor, "..."], expected: Shaped[Tensor, "..."]
) -> None:
    if value.layout != torch.strided or value.is_meta or expected.is_meta:
        message = f"Checkpoint loading requires materialized strided tensors: {key}"
        raise ValueError(message)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        message = (
            f"Checkpoint shape/dtype mismatch for {key}: "
            f"expected {tuple(expected.shape)} {expected.dtype}, "
            f"got {tuple(value.shape)} {value.dtype}"
        )
        raise ValueError(message)


def load_model_weights(
    path: str | Path,
    *,
    model: nn.Module,
    expected_sha256: str | None = None,
) -> CheckpointLoadReport:
    """Validate and load a plain ``model.state_dict()`` saved with ``torch.save``.

    Keys, tensor shapes, and dtypes must match exactly; validation precedes
    parameter changes. Official, EMA, optimizer, and resume envelopes are not
    accepted. Supply ``expected_sha256`` to additionally verify the file digest.
    The actual digest is always returned. Convert official weights with
    ``clean-rtdetr-convert`` first.
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        message = f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        raise ValueError(message)
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        message = "Checkpoint must be a plain state_dict mapping string keys to tensors"
        raise ValueError(message)
    expected = model.state_dict()
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        message = (
            f"Checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
        raise ValueError(message)
    for key, value in state.items():
        _validate_tensor(key, value, expected[key])
    model.load_state_dict(state, strict=True)
    return CheckpointLoadReport(digest, tuple(state))
