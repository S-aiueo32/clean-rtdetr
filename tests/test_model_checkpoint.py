# Copyright 2026 Clean RT-DETR contributors
"""Offline full-model checkpoint mapping and atomic validation tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pytest
import torch

from clean_rtdetr import build_rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from tests.official_oracle import backbone_oracle, full_transformer_oracle, hybrid_encoder_oracle

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def official_state() -> dict[str, torch.Tensor]:
    """Build independent upstream namespaces without downloading trained weights."""
    modules = (
        (
            "backbone",
            backbone_oracle(depth=50, variant="d", freeze_at=0, freeze_norm=True, pretrained=False),
        ),
        ("encoder", hybrid_encoder_oracle()),
        ("decoder", full_transformer_oracle(feat_channels=[256, 256, 256])),
    )
    return {
        f"{prefix}.{key}": value
        for prefix, module in modules
        for key, value in module.state_dict().items()
    }


def _save(
    tmp_path: Path, state: dict[str, torch.Tensor], source: Literal["model", "ema"]
) -> tuple[Path, str]:
    path = tmp_path / "weights.pth"
    torch.save({"ema": {"module": state}} if source == "ema" else {"model": state}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("source", ["model", "ema"])
def test_full_model_maps_every_inference_tensor(
    tmp_path: Path,
    official_state: dict[str, torch.Tensor],
    source: Literal["model", "ema"],
) -> None:
    """Consume backbone/encoder keys and synthesize only absent registered BN counters."""
    state = {
        key: value
        for key, value in official_state.items()
        if not key.endswith("num_batches_tracked")
    }
    path, digest = _save(tmp_path, state, source)
    model = build_rtdetr_r50vd()
    report = load_official_model_weights(path, model=model, source=source, expected_sha256=digest)
    assert report.source == source
    assert report.sha256 == digest
    assert set(report.excluded_keys) == {"decoder.denoising_class_embed.weight"}
    assert "encoder.input_proj.0.1.num_batches_tracked" in report.synthesized_keys
    assert "projector.input_proj.0.norm.num_batches_tracked" in report.synthesized_keys
    assert not any(key.startswith("backbone.") for key in report.synthesized_keys)
    loaded = model.state_dict()
    for upstream, target in report.key_mapping.items():
        torch.testing.assert_close(loaded[target], state[upstream], atol=0, rtol=0)
    for key in report.synthesized_keys:
        assert loaded[key].item() == 0


@pytest.mark.parametrize(
    "mutation", ["missing", "extra_backbone", "extra_encoder", "shape", "dtype", "hash", "source"]
)
def test_full_model_failure_does_not_change_any_stage(
    tmp_path: Path,
    official_state: dict[str, torch.Tensor],
    mutation: str,
) -> None:
    """Reject malformed full-model states before changing even the first backbone tensor."""
    state = dict(official_state)
    key = "encoder.pan_blocks.1.conv2.conv.weight"
    if mutation == "missing":
        state.pop(key)
    elif mutation.startswith("extra_"):
        state[mutation.removeprefix("extra_") + ".unknown.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = torch.zeros(1)
    elif mutation == "dtype":
        state[key] = state[key].double()
    path, digest = _save(tmp_path, state, "model")
    model = build_rtdetr_r50vd()
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match=r"checkpoint|Checkpoint"):
        load_official_model_weights(
            path,
            model=model,
            source="ema" if mutation == "source" else "model",
            expected_sha256="0" * 64 if mutation == "hash" else digest,
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)


def test_full_loader_rejects_same_shape_nonbaseline_attention(tmp_path: Path) -> None:
    """Changing attention heads changes semantics without changing checkpoint tensor shapes."""
    model = build_rtdetr_r50vd()
    model.decoder.config = replace(model.decoder.config, num_heads=4)
    with pytest.raises(ValueError, match="default R50vd"):
        load_official_model_weights(
            tmp_path / "not-read.pth",
            model=model,
            source="ema",
            expected_sha256="0" * 64,
        )
