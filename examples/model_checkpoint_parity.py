# Copyright 2026 Clean RT-DETR contributors
"""Compare all image-to-prediction stages using the pinned official EMA weights.

Run from the repository root. The independent upstream oracle lives in tests,
and the checkpoint remains an external, hash-verified artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from clean_rtdetr import build_rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from tests.official_oracle import (
    backbone_oracle,
    full_transformer_oracle,
    hybrid_encoder_oracle,
)

SEED = 41
ATOL = 1e-5
RTOL = 1e-4


def _compare(name: str, actual: Tensor, expected: Tensor) -> dict[str, object]:
    exact = name == "indices"
    torch.testing.assert_close(
        actual,
        expected,
        atol=0 if exact else ATOL,
        rtol=0 if exact else RTOL,
    )
    result: dict[str, object] = {"name": name, "shape": list(actual.shape), "exact": exact}
    if not exact:
        # Invalid anchors deliberately produce +inf reference logits.
        finite = torch.isfinite(expected)
        difference = (actual[finite] - expected[finite]).abs()
        result["max_abs"] = float(difference.max()) if difference.numel() else 0.0
        result["max_rel"] = (
            float((difference / expected[finite].abs().clamp_min(1e-12)).max())
            if difference.numel()
            else 0.0
        )
    return result


def _git_revision() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()  # noqa: S607
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_hashes() -> dict[str, str]:
    """Identify the actual implementation even when validation precedes a commit."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/clean_rtdetr").rglob("*.py")),
        *sorted((root / "third_party/RT-DETR").rglob("*.py")),
        root / "tests/oracle.py",
        root / "tests/official_oracle.py",
        Path(__file__).resolve(),
        root / "uv.lock",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


@torch.inference_mode()
def run(
    checkpoint: Path,
    expected_sha256: str,
    source: Literal["model", "ema"],
) -> dict[str, object]:
    """Validate the full FP32 640-square path and report every stage comparison."""
    torch.manual_seed(SEED)
    model = build_rtdetr_r50vd().eval()
    report = load_official_model_weights(
        checkpoint,
        model=model,
        source=source,
        expected_sha256=expected_sha256,
    )
    backbone = backbone_oracle(
        depth=50,
        variant="d",
        freeze_at=0,
        return_idx=[1, 2, 3],
        num_stages=4,
        freeze_norm=True,
        pretrained=False,
    ).eval()
    encoder = cast("Any", hybrid_encoder_oracle(eval_spatial_size=[640, 640])).eval()
    transformer = cast(
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload[source]["module"] if source == "ema" else payload[source]
    for prefix, module in (("backbone", backbone), ("encoder", encoder), ("decoder", transformer)):
        module.load_state_dict(
            {
                key.removeprefix(prefix + "."): value
                for key, value in state.items()
                if key.startswith(prefix + ".")
            },
            strict=True,
        )

    images = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(SEED))
    clean_features = model.backbone(images)
    oracle_features = backbone(images)
    checks = [
        _compare(f"backbone.{index}", actual, expected)
        for index, (actual, expected) in enumerate(
            zip(clean_features.levels, oracle_features, strict=True)
        )
    ]
    captured: list[Tensor] = []

    def capture_aifi(_module: nn.Module, _args: tuple[object, ...], output: Tensor) -> None:
        captured.append(output)

    handle = encoder.encoder[0].register_forward_hook(capture_aifi)
    try:
        oracle_fused = encoder(oracle_features)
    finally:
        handle.remove()
    clean_encoded = model.encoder(clean_features, collect_aifi=True)
    if clean_encoded.after_aifi is None:
        msg = "Requested AIFI feature was not returned"
        raise AssertionError(msg)
    checks.append(
        _compare(
            "after_aifi",
            clean_encoded.after_aifi.levels[0],
            captured[0].transpose(1, 2).reshape(1, 256, 20, 20),
        )
    )
    checks.extend(
        _compare(f"encoder.{index}", actual, expected)
        for index, (actual, expected) in enumerate(
            zip(clean_encoded.features.levels, oracle_fused, strict=True)
        )
    )
    memory = model.projector(clean_encoded.features)
    selected = model.selector(memory)
    queries = model.assembler(selected.queries)
    output = model.decoder(queries, memory)
    oracle_memory, shapes, offsets = transformer._get_encoder_input(oracle_fused)  # noqa: SLF001
    target, refs, enc_boxes, enc_logits = transformer._get_decoder_input(oracle_memory, shapes)  # noqa: SLF001
    _, valid = transformer._generate_anchors(shapes)  # noqa: SLF001
    scores = transformer.enc_score_head(
        transformer.enc_output(oracle_memory * valid.to(oracle_memory.dtype))
    )
    indices = scores.max(-1).values.topk(300, dim=1).indices
    for name, actual, expected in (
        ("memory", memory.memory, oracle_memory),
        ("indices", selected.queries.source_indices, indices),
        ("query", queries.content, target),
        ("references", queries.ref_logits, refs),
        ("encoder_logits", selected.encoder_logits, enc_logits),
        ("encoder_boxes", selected.encoder_boxes, enc_boxes),
    ):
        checks.append(_compare(name, actual, expected))
    for index in range(6):
        transformer.decoder.eval_idx = index
        boxes, logits = transformer.decoder(
            target,
            refs,
            oracle_memory,
            shapes,
            offsets,
            transformer.dec_bbox_head,
            transformer.dec_score_head,
            transformer.query_pos_head,
        )
        checks.append(_compare(f"decoder.{index}.logits", output.logits[index], logits[-1]))
        checks.append(_compare(f"decoder.{index}.boxes", output.boxes[index], boxes[-1]))
    composed = model(images)
    checks.append(_compare("composed.logits", composed.decoder.logits, output.logits))
    checks.append(_compare("composed.boxes", composed.decoder.boxes, output.boxes))
    return {
        "checkpoint": str(checkpoint),
        "sha256": report.sha256,
        "source": report.source,
        "key_mapping": report.key_mapping,
        "excluded_keys": report.excluded_keys,
        "synthesized_keys": report.synthesized_keys,
        "checks": checks,
        "seed": SEED,
        "image_shape": list(images.shape),
        "input": "RGB float [0,1], no normalization",
        "atol": ATOL,
        "rtol": RTOL,
        "git_revision": _git_revision(),
        "source_sha256": _source_hashes(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(images.device),
        "dtype": str(images.dtype),
        "torch_num_threads": torch.get_num_threads(),
    }


def main() -> None:
    """Read explicit artifact identity and write a reproducible parity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source", choices=("model", "ema"), default="ema")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run(
        args.checkpoint, args.expected_sha256, cast("Literal['model', 'ema']", args.source)
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.report:
        args.report.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
