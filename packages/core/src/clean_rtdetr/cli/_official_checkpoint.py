# Copyright 2026 Clean RT-DETR contributors
"""Load official RT-DETR model weights for the private conversion CLI."""

import hashlib
import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Shaped
from torch import Tensor, nn

from clean_rtdetr.models.backbone import FrozenBatchNorm2d, PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderLayer
from clean_rtdetr.models.encoder import HybridEncoder
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.rtdetr import BaselineRTDETR
from clean_rtdetr.models.selector import BoxHead, QuerySelector


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Record the selected artifact and every mapped, excluded, or supplied key."""

    sha256: str
    source: Literal["model", "ema"]
    key_mapping: dict[str, str]
    excluded_keys: dict[str, str]
    synthesized_keys: tuple[str, ...]


def _validate_full_model(model: BaselineRTDETR) -> None:
    """Restrict whole-checkpoint conversion to supported explicit stage types."""
    if not all(
        type(getattr(model, name, None)) is expected
        for name, expected in (
            ("backbone", PResNet),
            ("encoder", HybridEncoder),
            ("projector", MemoryProjector),
            ("selector", QuerySelector),
            ("assembler", QueryAssembler),
            ("decoder", Decoder),
        )
    ):
        msg = "Official conversion requires the default RT-DETR stage modules"
        raise ValueError(msg)
    _validate_modules(model.projector, model.selector, model.decoder)
    if tuple(model.backbone.specs) != tuple(model.encoder.specs):
        msg = "Backbone and encoder feature specs are incompatible"
        raise ValueError(msg)
    if (
        model.encoder.output_specs != model.projector.specs
        or model.encoder.hidden_dim != model.projector.hidden_dim
    ):
        msg = "Encoder and projector feature specs are incompatible"
        raise ValueError(msg)


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


def _state_dict(
    payload: object, source: Literal["model", "ema"]
) -> Mapping[str, Shaped[Tensor, "..."]]:
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


def _full_source_key(stage: str, target: str) -> str:
    """Translate a local model key to the official checkpoint namespace."""
    if stage in {"backbone", "encoder"}:
        return f"{stage}.{target}"
    return _source_key(stage, target)


def _is_frozen_bn_counter(model: nn.Module, key: str) -> bool:
    """Recognize only counters belonging to local FrozenBatchNorm modules."""
    if not key.endswith(".num_batches_tracked"):
        return False
    try:
        module = model.get_submodule(key.removesuffix(".num_batches_tracked"))
    except AttributeError:
        return False
    return isinstance(module, FrozenBatchNorm2d)


def _is_batch_norm_counter(module: nn.Module, target: str) -> bool:
    """Recognize counters only when their destination is a real BatchNorm."""
    if not target.endswith(".num_batches_tracked"):
        return False
    try:
        destination = module.get_submodule(target.removesuffix(".num_batches_tracked"))
    except AttributeError:
        return False
    return isinstance(
        destination, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    )


def _validate_tensor(
    key: str, value: Shaped[Tensor, "..."], expected: Shaped[Tensor, "..."]
) -> None:
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


def _convert_modules(
    state: Mapping[str, Shaped[Tensor, "..."]],
    modules: Mapping[str, nn.Module],
    source_key: Callable[[str, str], str],
) -> tuple[dict[str, dict[str, Shaped[Tensor, "..."]]], dict[str, str], list[str]]:
    """Validate and convert module state dictionaries without mutating modules."""
    converted: dict[str, dict[str, Tensor]] = {}
    mapping: dict[str, str] = {}
    synthesized: list[str] = []
    for stage, module in modules.items():
        converted[stage] = {}
        for target, expected in module.state_dict().items():
            key = source_key(stage, target)
            destination = f"{stage}.{target}"
            if key in mapping:
                msg = f"Duplicate checkpoint key mapping: {key}"
                raise ValueError(msg)
            if key not in state:
                if _is_batch_norm_counter(module, target):
                    converted[stage][target] = torch.zeros_like(expected)
                    synthesized.append(destination)
                    continue
                msg = f"Missing checkpoint key: {key}"
                raise ValueError(msg)
            value = state[key]
            _validate_tensor(key, value, expected)
            converted[stage][target] = value
            mapping[key] = destination
    return converted, mapping, synthesized


def load_official_model_weights(
    path: str | Path,
    *,
    model: BaselineRTDETR,
    source: Literal["model", "ema"],
    expected_sha256: str,
) -> CheckpointLoadReport:
    """Validate then load all inference stages of a supported preset composition.

    The model/EMA source and SHA-256 are mandatory. All target shapes and dtypes
    must match; missing BatchNorm counters are zero-initialized and reported.
    The training-only DN embedding is excluded. Optimizer/resume state and
    custom stages are unsupported.
    """
    _validate_full_model(model)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        msg = f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        raise ValueError(msg)
    state = _state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), source)
    modules = {
        "backbone": model.backbone,
        "encoder": model.encoder,
        "projector": model.projector,
        "selector": model.selector,
        "decoder": model.decoder,
    }
    converted, mapping, synthesized = _convert_modules(state, modules, _full_source_key)
    excluded: dict[str, str] = {}
    for key in state.keys() - mapping.keys():
        if _is_frozen_bn_counter(model, key):
            excluded[key] = "FrozenBatchNorm counter is not part of the target module"
        elif key == "decoder.denoising_class_embed.weight":
            excluded[key] = "Training denoising generator is not configured"
        elif key in {"decoder.anchors", "decoder.valid_mask"}:
            excluded[key] = "Fixed-size inference geometry is generated from runtime features"
        else:
            msg = f"Unexpected checkpoint key: {key}"
            raise ValueError(msg)
    for stage, module in modules.items():
        module.load_state_dict(converted[stage], strict=True)
    return CheckpointLoadReport(digest, source, mapping, excluded, tuple(synthesized))
