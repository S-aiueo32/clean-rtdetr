# Copyright 2026 Clean RT-DETR contributors
"""Offline tests for official RT-DETR stage checkpoint loading."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal, cast

import pytest
import torch
from torch import nn

from clean_rtdetr import (
    Decoder,
    DecoderConfig,
    FeatureSpec,
    MemoryProjector,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr.compatibility import CheckpointLoadReport, load_official_stage_weights

if TYPE_CHECKING:
    from pathlib import Path

BN_MARKER = 17


def _modules() -> tuple[MemoryProjector, QuerySelector, Decoder]:
    """Build a small baseline with every supported stage parameter present."""
    specs = (FeatureSpec(3, 8), FeatureSpec(5, 16))
    projector = MemoryProjector(specs, hidden_dim=8)
    selector = QuerySelector(
        QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=4, learned_content=True)
    )
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=8,
            num_classes=3,
            num_layers=2,
            num_heads=2,
            num_levels=2,
            num_points=2,
            feedforward_dim=16,
        )
    )
    return projector, selector, decoder


def _official_state(
    projector: nn.Module, selector: nn.Module, decoder: nn.Module
) -> dict[str, torch.Tensor]:
    """Create an upstream-shaped state dict from independently named modules."""
    state: dict[str, torch.Tensor] = {}
    for prefix, module in (("projector", projector), ("selector", selector), ("decoder", decoder)):
        for key, value in module.state_dict().items():
            if prefix == "projector":
                source = "decoder.input_proj." + key.removeprefix("input_proj.")
            elif prefix == "selector":
                source = "decoder." + key
            else:
                source = "decoder.decoder." + key if key.startswith("layers.") else "decoder." + key
            state[source] = value.detach().clone()
    state["backbone.conv.weight"] = torch.randn(2, 2)
    state["encoder.unused.weight"] = torch.randn(2)
    state["decoder.denoising_class_embed.weight"] = torch.randn(7, 8)
    return state


def _save(
    tmp_path: Path, state: dict[str, torch.Tensor], *, source: Literal["model", "ema"] = "model"
) -> Path:
    path = tmp_path / f"{source}.pth"
    payload: object = {source: state}
    if source == "ema":
        payload = {"ema": {"module": state}}
    torch.save(payload, path)
    return path


def _load(
    path: Path,
    *,
    source: Literal["model", "ema"] = "model",
    expected_sha256: str | None = None,
) -> tuple[CheckpointLoadReport, MemoryProjector, QuerySelector, Decoder]:
    projector, selector, decoder = _modules()
    if expected_sha256 is None:
        expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    report = load_official_stage_weights(
        path,
        projector=projector,
        selector=selector,
        decoder=decoder,
        source=source,
        expected_sha256=expected_sha256,
    )
    return report, projector, selector, decoder


def test_loads_model_and_reports_mapping_exclusions_and_bn_synthesis(tmp_path: Path) -> None:
    """Load the model branch and expose mappings, exclusions, and synthesized buffers."""
    source_projector, source_selector, source_decoder = _modules()
    state = _official_state(source_projector, source_selector, source_decoder)
    state["decoder.input_proj.1.norm.num_batches_tracked"] = torch.tensor(
        BN_MARKER, dtype=torch.int64
    )
    state.pop("decoder.input_proj.0.norm.num_batches_tracked")
    path = _save(tmp_path, state)
    report, projector, selector, decoder = _load(path)

    assert report.source == "model"
    assert report.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        report.key_mapping["decoder.input_proj.0.conv.weight"]
        == "projector.input_proj.0.conv.weight"
    )
    assert report.excluded_keys["backbone.conv.weight"]
    assert report.excluded_keys["encoder.unused.weight"]
    assert report.excluded_keys["decoder.denoising_class_embed.weight"]
    assert "projector.input_proj.0.norm.num_batches_tracked" in report.synthesized_keys
    norm = cast("torch.nn.BatchNorm2d", projector.input_proj[1].norm)
    assert norm.num_batches_tracked is not None
    assert norm.num_batches_tracked.item() == BN_MARKER
    for module in (projector, selector, decoder):
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())


def test_ema_module_is_selected_explicitly(tmp_path: Path) -> None:
    """Select EMA weights only when the explicit EMA source is requested."""
    projector, selector, decoder = _modules()
    state = _official_state(projector, selector, decoder)
    marker = torch.full_like(state["decoder.enc_score_head.weight"], 3.0)
    state["decoder.enc_score_head.weight"] = marker
    path = _save(tmp_path, state, source="ema")
    report, _, loaded_selector, _ = _load(path, source="ema")
    assert report.source == "ema"
    torch.testing.assert_close(loaded_selector.enc_score_head.weight, marker)


@pytest.mark.parametrize("bad_key", ["decoder.unknown.weight", "decoder.input_proj.9.conv.weight"])
def test_unknown_or_missing_keys_fail_atomically(tmp_path: Path, bad_key: str) -> None:
    """Reject unknown and missing source keys without changing any destination module."""
    projector, selector, decoder = _modules()
    state = _official_state(projector, selector, decoder)
    if bad_key.startswith("decoder.unknown"):
        state[bad_key] = torch.randn(2)
    else:
        state.pop("decoder.input_proj.0.conv.weight")
    path = _save(tmp_path, state)
    before = [
        {key: value.detach().clone() for key, value in module.state_dict().items()}
        for module in (projector, selector, decoder)
    ]
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        load_official_stage_weights(
            path,
            projector=projector,
            selector=selector,
            decoder=decoder,
            source="model",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    for module, snapshot in zip((projector, selector, decoder), before, strict=True):
        assert all(torch.equal(module.state_dict()[key], value) for key, value in snapshot.items())


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_invalid_tensor_metadata_fails_without_partial_load(tmp_path: Path, mutation: str) -> None:
    """Reject shape and dtype mismatches before mutating destination modules."""
    projector, selector, decoder = _modules()
    state = _official_state(projector, selector, decoder)
    key = "decoder.enc_score_head.weight"
    original = state[key]
    state[key] = torch.randn(1) if mutation == "shape" else original.double()
    path = _save(tmp_path, state)
    before = [
        {key: value.detach().clone() for key, value in module.state_dict().items()}
        for module in (projector, selector, decoder)
    ]
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        load_official_stage_weights(
            path,
            projector=projector,
            selector=selector,
            decoder=decoder,
            source="model",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    for module, snapshot in zip((projector, selector, decoder), before, strict=True):
        assert all(torch.equal(module.state_dict()[key], value) for key, value in snapshot.items())


def test_sha256_and_source_are_strict(tmp_path: Path) -> None:
    """Require the expected digest and the requested checkpoint branch."""
    projector, selector, decoder = _modules()
    path = _save(tmp_path, _official_state(projector, selector, decoder))
    with pytest.raises(ValueError, match="SHA-256"):
        _load(path, expected_sha256="0" * 64)
    with pytest.raises((KeyError, ValueError)):
        _load(path, source="ema")

    malformed = tmp_path / "malformed-ema.pth"
    torch.save({"ema": {"weights": _official_state(*_modules())}}, malformed)
    with pytest.raises(ValueError, match="module"):
        _load(malformed, source="ema")


def test_custom_heads_and_configuration_mismatch_are_rejected(tmp_path: Path) -> None:
    """Reject custom heads and decoder configurations outside the conversion contract."""
    source_projector, source_selector, source_decoder = _modules()
    path = _save(tmp_path, _official_state(source_projector, source_selector, source_decoder))
    projector = MemoryProjector((FeatureSpec(3, 8), FeatureSpec(5, 16)), hidden_dim=8)
    selector = QuerySelector(
        QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=4),
        score_head=nn.Sequential(nn.Linear(8, 3)),
    )
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=8,
            num_classes=3,
            num_layers=1,
            num_heads=2,
            num_levels=2,
            num_points=2,
            feedforward_dim=16,
        )
    )
    with pytest.raises((TypeError, ValueError, KeyError)):
        load_official_stage_weights(
            path,
            projector=projector,
            selector=selector,
            decoder=decoder,
            source="model",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
