# Copyright 2026 Clean RT-DETR contributors
# ruff: noqa: TRY003, TRY301, EM101, EM102
"""Attempt a checkout-only native CUDA FP16 ONNX export for deployment validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr_export import FixedShapeRTDETR
from tools.compatibility import load_official_model_weights

CHECKPOINT_SHA256 = "dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45"


class _ExperimentalCudaFP16(FixedShapeRTDETR):
    """Relax the public adapter boundary only for this bounded experiment."""

    def _validate_input(self, images: torch.Tensor) -> None:
        if images.shape != self.input_shape:
            raise ValueError(f"Expected fixed input shape {self.input_shape}")
        if images.dtype != torch.float16 or images.device.type != "cuda":
            raise ValueError("Experimental export expects CUDA float16 input")
        if self.training or any(module.training for module in self.model.modules()):
            raise ValueError("Experimental export requires eval mode")


def main() -> None:
    """Export native-half weights and always persist the attempt outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {
        "status": "failed",
        "scope": "checkout-only experimental CUDA FP16; public adapter remains CPU FP32",
    }
    try:
        model = rtdetr_r50vd().eval()
        load_official_model_weights(
            args.checkpoint, model=model, source="ema", expected_sha256=CHECKPOINT_SHA256
        )
        adapter = _ExperimentalCudaFP16(model)
        fp32_anchors = adapter.anchors.clone()
        adapter = adapter.cuda().half().eval()
        adapter.anchors = fp32_anchors.cuda()
        sample = torch.rand(1, 3, 640, 640, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            program = torch.onnx.export(
                adapter,
                (sample,),
                dynamo=True,
                opset_version=18,
                input_names=["images"],
                output_names=["logits", "boxes"],
                external_data=False,
                optimize=False,
            )
        if program is None:
            raise RuntimeError("Dynamo exporter returned no ONNX program")
        program.save(args.output, external_data=False)
        report.update(
            status="exported",
            artifact=str(args.output),
            artifact_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
            input_dtype="float16",
            weights_dtype="float16",
            anchors_dtype=str(adapter.anchors.dtype),
        )
    except Exception as error:  # noqa: BLE001 - experiment must retain unsupported outcome
        report.update(error=str(error), traceback=traceback.format_exc())
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "exported":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
