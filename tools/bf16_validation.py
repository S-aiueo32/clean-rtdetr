# Copyright 2026 Clean RT-DETR contributors
"""Run native CUDA BF16 regressions and full R50vd official-weight inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights


def main() -> None:
    """Require native hardware, run regressions, and preserve bounded evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("A native BF16 CUDA GPU is required; emulation is not accepted")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    junit = args.output.resolve() / "regressions.xml"
    with (args.output / "regressions.log").open("w") as log:
        subprocess.run(  # noqa: S603 - fixed interpreter and checkout test paths
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "packages/core/tests/test_precision.py",
                "packages/training/tests/test_gpu_precision.py",
                f"--junitxml={junit}",
            ],
            cwd=root,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    suite = ET.parse(junit).getroot()[0]  # noqa: S314
    if any(int(suite.attrib[key]) for key in ("failures", "errors", "skipped")):
        message = "Native GPU regression tests must all execute and pass"
        raise RuntimeError(message)
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = rtdetr_r50vd().eval()
    load_model_weights(args.checkpoint, model=model, expected_sha256=args.expected_sha256)
    model.cuda()
    image = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.inference_mode():
        fp32 = model(image).final
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = model(image).final
    if bf16.logits.dtype != torch.float32 or bf16.boxes.dtype != torch.float32:
        message = "BF16 autocast must preserve FP32 public predictions"
        raise RuntimeError(message)
    if not torch.isfinite(bf16.logits).all() or not torch.isfinite(bf16.boxes).all():
        message = "Official-weight BF16 inference produced nonfinite predictions"
        raise RuntimeError(message)
    report = {
        "status": "passed_native_bf16",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "native_bf16_supported": True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "seed": 41,
        "checkpoint_sha256": args.expected_sha256,
        "shape": list(image.shape),
        "public_predictions_dtype": "float32",
        "pytest": suite.attrib,
        "fp32_comparison_ordered_diagnostic": {
            "max_abs_logits": (bf16.logits - fp32.logits).abs().max().item(),
            "max_abs_boxes": (bf16.boxes - fp32.boxes).abs().max().item(),
            "note": "Query selection can differ; ordered differences do not assert parity or AP.",
        },
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
            if p.is_file()
        },
        "scope": (
            "Native BF16 inference and compact DN updates; official full R50vd 640 inference. "
            "No BF16 AP, DDP, Lightning BF16, or convergence claim."
        ),
    }
    (args.output / "bf16-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
