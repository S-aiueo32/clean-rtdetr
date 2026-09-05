# Copyright 2026 Clean RT-DETR contributors
"""Compare clean stages with a pinned official RT-DETR checkpoint.

Run from the repository root after obtaining the official checkpoint.  The
oracle is intentionally imported from ``tests`` so it cannot become installed
package code or silently replace the clean implementation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import torch

from clean_rtdetr import (
    Decoder,
    DecoderConfig,
    FeatureSpec,
    MemoryProjector,
    MultiScaleFeatures,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr.compatibility import load_official_stage_weights
from tests.official_oracle import full_transformer_oracle

SEED = 41
ATOL = 1e-5
RTOL = 1e-4


def _git_revision() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()  # noqa: S607
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
) -> dict[str, object]:
    """Run deterministic stage parity against one official checkpoint."""
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
        "key_mapping": report.key_mapping,
        "excluded_keys": report.excluded_keys,
        "checks": checks,
        "seed": SEED,
        "atol": ATOL,
        "rtol": RTOL,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    return result  # noqa: RET504


def main() -> None:
    """Parse command line arguments and emit the parity report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source", choices=("model", "ema"), default="model")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run(
        args.checkpoint,
        args.expected_sha256,
        cast("Literal['model', 'ema']", args.source),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.report:
        args.report.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
