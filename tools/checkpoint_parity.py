# Copyright 2026 Clean RT-DETR contributors
"""Compare clean stages with a pinned official RT-DETR checkpoint.

Run from the repository root after obtaining the official checkpoint.  The
oracle is intentionally imported from ``packages/core/tests`` so it cannot
become installed package code or silently replace the clean implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

import torch

# Direct script execution puts tools/ on sys.path; the parity-only oracle lives
# in the core package's tests rather than the installed library.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "packages/core/tests"))

from official_oracle import (
    BACKBONE_COMMON_SHA256,
    HYBRID_ENCODER_SHA256,
    PRESNET_SHA256,
    UPSTREAM_COMMIT,
    UTILS_SHA256,
    backbone_oracle,
    full_transformer_oracle,
    hybrid_encoder_oracle,
)
from oracle import SHA256 as DECODER_ORACLE_SHA256

from clean_rtdetr import (
    Decoder,
    DecoderConfig,
    FeatureSpec,
    MemoryProjector,
    MultiScaleFeatures,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
    rtdetr_r50vd,
)
from tools.compatibility import load_official_model_weights, load_official_stage_weights

SEED = 41
ATOL = 1e-5
RTOL = 1e-4
IMPLEMENTATION_PATHS = (
    "tools/checkpoint_parity.py",
    "packages/core/tests/oracle.py",
    "packages/core/tests/official_oracle.py",
)


def _git_revision() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()  # noqa: S607
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _worktree_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"),  # noqa: S607
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def _implementation_hashes() -> dict[str, str]:
    """Hash implementation and oracle sources included in a parity run."""
    paths = [
        *Path("packages/core/src/clean_rtdetr").rglob("*.py"),
        *Path("tools/compatibility").rglob("*.py"),
        *(Path(path) for path in IMPLEMENTATION_PATHS),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if name == "indices":
        equal = torch.equal(actual, expected)
        if not equal:
            msg = f"{name} mismatch"
            raise AssertionError(msg)
        return {"name": name, "exact": True, "shape": list(actual.shape)}
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    difference = (actual - expected).abs()
    return {
        "name": name,
        "exact": False,
        "shape": list(actual.shape),
        "max_abs": float(difference.max()),
        "max_rel": float((difference / expected.abs().clamp_min(1e-12)).max()),
    }


@torch.inference_mode()
def run(
    checkpoint: Path,
    expected_sha256: str,
    source: Literal["model", "ema"],
    full_model: bool = False,  # noqa: FBT001, FBT002
) -> dict[str, object]:
    """Run deterministic stage parity against one official checkpoint."""
    if full_model:
        return _run_full_model(checkpoint, expected_sha256, source)
    torch.manual_seed(SEED)
    specs = (FeatureSpec(256, 8), FeatureSpec(256, 16), FeatureSpec(256, 32))
    config = DecoderConfig(num_levels=3)
    projector = MemoryProjector(specs, hidden_dim=256).eval()
    selector = QuerySelector(QuerySelectorConfig(learned_content=False)).eval()
    decoder = Decoder(config).eval()
    report = load_official_stage_weights(
        checkpoint,
        projector=projector,
        selector=selector,
        decoder=decoder,
        source=source,
        expected_sha256=expected_sha256,
    )

    levels = tuple(
        torch.randn(1, spec.channels, height, height)
        for spec, height in zip(specs, (80, 40, 20), strict=True)
    )
    encoded = projector(MultiScaleFeatures(levels, specs))
    selected = selector(encoded)
    queries = QueryAssembler()(selected.queries)
    clean_output = decoder(queries, encoded)

    oracle = cast(
        "Any",
        full_transformer_oracle(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.0,
            activation="relu",
            learnt_init_query=False,
            num_denoising=100,
        ),
    ).eval()
    # The conversion report contains clean keys; the oracle receives the
    # original official state dict through the compatibility loader's source.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or source not in state:
        msg = f"Checkpoint has no {source!r} state"
        raise ValueError(msg)
    state = state[source]
    if source == "ema":
        if not isinstance(state, dict) or "module" not in state:
            msg = "Checkpoint EMA has no module state"
            raise ValueError(msg)
        state = state["module"]
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    oracle.load_state_dict(decoder_state, strict=True)
    memory, shapes, offsets = oracle._get_encoder_input(levels)  # noqa: SLF001
    target, refs, enc_boxes, enc_logits = oracle._get_decoder_input(  # noqa: SLF001
        memory, shapes
    )
    _, valid = oracle._generate_anchors(shapes)  # noqa: SLF001
    oracle_encoded = oracle.enc_output(memory * valid.to(memory.dtype))
    oracle_selection_logits = oracle.enc_score_head(oracle_encoded)
    oracle_indices = oracle_selection_logits.max(-1).values.topk(300, dim=1).indices
    oracle_boxes_by_layer = []
    oracle_logits_by_layer = []
    for index in range(6):
        oracle.decoder.eval_idx = index
        boxes, logits = oracle.decoder(
            target,
            refs,
            memory,
            shapes,
            offsets,
            oracle.dec_bbox_head,
            oracle.dec_score_head,
            oracle.query_pos_head,
        )
        oracle_boxes_by_layer.append(boxes[-1])
        oracle_logits_by_layer.append(logits[-1])
    oracle_boxes = torch.stack(oracle_boxes_by_layer)
    oracle_logits = torch.stack(oracle_logits_by_layer)
    checks = [
        _compare("memory", encoded.memory, memory),
        _compare(
            "indices",
            selected.queries.source_indices,
            oracle_indices,
        ),
        _compare("query", queries.content, target),
        _compare("references", queries.ref_logits, refs),
        _compare("encoder_boxes", selected.encoder_boxes, enc_boxes),
        _compare("encoder_logits", selected.encoder_logits, enc_logits),
        _compare("logits", clean_output.logits, oracle_logits),
        _compare("boxes", clean_output.boxes, oracle_boxes),
    ]
    result = {
        "checkpoint": str(checkpoint),
        "sha256": report.sha256,
        "source": report.source,
        "key_mapping": report.key_mapping,
        "excluded_keys": report.excluded_keys,
        "synthesized_keys": report.synthesized_keys,
        "checks": checks,
        "seed": SEED,
        "atol": ATOL,
        "rtol": RTOL,
        "git_revision": _git_revision(),
        "worktree_dirty": _worktree_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
    }
    return result  # noqa: RET504


@torch.inference_mode()
def _run_full_model(  # noqa: C901, PLR0915
    checkpoint: Path,
    expected_sha256: str,
    source: Literal["model", "ema"],
) -> dict[str, object]:
    """Compare the composed image-to-prediction model with independent upstream stages."""
    torch.manual_seed(SEED)
    model = rtdetr_r50vd().eval()
    report = load_official_model_weights(
        checkpoint,
        model=model,
        source=source,
        expected_sha256=expected_sha256,
    )
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randn((1, 3, 640, 640), generator=generator, dtype=torch.float32)
    wrapped = model(images, collect_aifi=True)
    if wrapped.after_aifi is None:
        msg = "Full-model parity requires wrapper AIFI features"
        raise AssertionError(msg)

    # Re-run the public stage boundaries to ensure the wrapper does not alter any
    # tensors or execution order.
    features = model.backbone(images)
    encoded_features = model.encoder(features, collect_aifi=True)
    if encoded_features.after_aifi is None:
        msg = "Full-model parity requires staged AIFI features"
        raise AssertionError(msg)
    encoded = model.projector(encoded_features.features)
    selected = model.selector(encoded)
    queries = model.assembler(selected.queries)
    manual = model.decoder(queries, encoded)

    oracle_backbone = backbone_oracle(
        depth=50,
        variant="d",
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        pretrained=False,
    ).eval()
    oracle_encoder = hybrid_encoder_oracle(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=[640, 640],
    ).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or source not in payload:
        msg = f"Checkpoint has no {source!r} state"
        raise ValueError(msg)
    state = payload[source]
    if source == "ema":
        if not isinstance(state, dict) or "module" not in state:
            msg = "Checkpoint EMA has no module state"
            raise ValueError(msg)
        state = state["module"]
    if not isinstance(state, dict):
        msg = "Selected checkpoint state is not a mapping"
        raise TypeError(msg)
    oracle_backbone.load_state_dict(
        {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        },
        strict=True,
    )
    oracle_encoder.load_state_dict(
        {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        },
        strict=True,
    )
    aifi_capture: list[torch.Tensor] = []
    hook = oracle_encoder.get_submodule("encoder.0").register_forward_hook(
        lambda _module, _inputs, output: aifi_capture.append(output)
    )
    oracle_levels = tuple(oracle_backbone(images))
    oracle_encoded_levels = tuple(oracle_encoder(list(oracle_levels)))
    hook.remove()
    if not aifi_capture:
        msg = "Full-model parity oracle did not capture AIFI output"
        raise AssertionError(msg)
    oracle_aifi = aifi_capture[0].transpose(1, 2).reshape(1, 256, 20, 20)
    oracle_transformer = cast(
        "Any",
        full_transformer_oracle(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.0,
            activation="relu",
            learnt_init_query=False,
            num_denoising=100,
            eval_spatial_size=[640, 640],
        ),
    ).eval()
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    oracle_transformer.load_state_dict(decoder_state, strict=True)
    oracle_memory_for_decoder, shapes, offsets = oracle_transformer._get_encoder_input(  # noqa: SLF001
        list(oracle_encoded_levels)
    )
    target, refs, oracle_enc_boxes, oracle_enc_logits = oracle_transformer._get_decoder_input(  # noqa: SLF001
        oracle_memory_for_decoder, shapes
    )
    _, valid = oracle_transformer._generate_anchors(shapes)  # noqa: SLF001
    oracle_encoded = oracle_transformer.enc_output(
        oracle_memory_for_decoder * valid.to(oracle_memory_for_decoder.dtype)
    )
    oracle_selection_logits = oracle_transformer.enc_score_head(oracle_encoded)
    oracle_indices = oracle_selection_logits.max(-1).values.topk(300, dim=1).indices
    oracle_boxes_by_layer = []
    oracle_logits_by_layer = []
    for index in range(6):
        oracle_transformer.decoder.eval_idx = index
        boxes, logits = oracle_transformer.decoder(
            target,
            refs,
            oracle_memory_for_decoder,
            shapes,
            offsets,
            oracle_transformer.dec_bbox_head,
            oracle_transformer.dec_score_head,
            oracle_transformer.query_pos_head,
        )
        oracle_boxes_by_layer.append(boxes[-1])
        oracle_logits_by_layer.append(logits[-1])
    oracle_boxes = torch.stack(oracle_boxes_by_layer)
    oracle_logits = torch.stack(oracle_logits_by_layer)
    checks: list[dict[str, object]] = [
        _compare("wrapper_decoder_logits", wrapped.decoder.logits, manual.logits),
        _compare("wrapper_decoder_boxes", wrapped.decoder.boxes, manual.boxes),
        _compare("wrapper_encoder_logits", wrapped.encoder_aux.logits, selected.encoder_logits),
        _compare("wrapper_encoder_boxes", wrapped.encoder_aux.boxes, selected.encoder_boxes),
        _compare(
            "wrapper_after_aifi",
            wrapped.after_aifi.levels[-1],
            encoded_features.after_aifi.levels[-1],
        ),
    ]
    for index, (actual, expected) in enumerate(zip(features.levels, oracle_levels, strict=True)):
        checks.append(_compare(f"backbone_level_{index}", actual, expected))
    for index, (actual, expected) in enumerate(
        zip(encoded_features.features.levels, oracle_encoded_levels, strict=True)
    ):
        checks.append(_compare(f"encoder_level_{index}", actual, expected))
    checks.extend(
        (
            _compare("after_aifi", encoded_features.after_aifi.levels[-1], oracle_aifi),
            _compare("memory", encoded.memory, oracle_memory_for_decoder),
        )
    )
    # The clean selector and decoder are loaded from the same checkpoint and
    # independently exercised by the staged parity path; compare every output
    # tensor here so full-model reports remain self-contained.
    checks.extend(
        (
            _compare("indices", selected.queries.source_indices, oracle_indices),
            _compare("query", queries.content, target),
            _compare("references", queries.ref_logits, refs),
            _compare("encoder_boxes", selected.encoder_boxes, oracle_enc_boxes),
            _compare("encoder_logits", selected.encoder_logits, oracle_enc_logits),
            _compare("logits", wrapped.decoder.logits, oracle_logits),
            _compare("boxes", wrapped.decoder.boxes, oracle_boxes),
        )
    )
    return {
        "checkpoint": str(checkpoint),
        "sha256": report.sha256,
        "source": source,
        "full_model": True,
        "key_mapping": report.key_mapping,
        "excluded_keys": report.excluded_keys,
        "synthesized_keys": report.synthesized_keys,
        "checks": checks,
        "seed": SEED,
        "input_shape": list(images.shape),
        "input_dtype": str(images.dtype),
        "input_sha256": hashlib.sha256(bytes(images.untyped_storage())).hexdigest(),
        "atol": ATOL,
        "rtol": RTOL,
        "git_revision": _git_revision(),
        "worktree_dirty": _worktree_dirty(),
        "upstream_commit": UPSTREAM_COMMIT,
        "oracle_source_sha256": {
            "backbone_common": BACKBONE_COMMON_SHA256,
            "presnet": PRESNET_SHA256,
            "hybrid_encoder": HYBRID_ENCODER_SHA256,
            "decoder": DECODER_ORACLE_SHA256,
            "utils": UTILS_SHA256,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "implementation_sha256": _implementation_hashes(),
    }


def main() -> None:
    """Parse command line arguments and emit the parity report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source", choices=("model", "ema"), default="model")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run(
        args.checkpoint,
        args.expected_sha256,
        cast("Literal['model', 'ema']", args.source),
        args.full_model,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.report:
        args.report.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
