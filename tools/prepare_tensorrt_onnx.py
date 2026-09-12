# Copyright 2026 Clean RT-DETR contributors
"""Prepare an ONNX copy for bounded TensorRT experiments and verify CPU equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxscript import optimizer


def _operators(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node)


def main() -> None:
    """Optimize a copy, validate it, and compare both graphs through ORT CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        message = "Output and report paths must not already exist"
        raise FileExistsError(message)

    original = onnx.load(args.input)
    before = _operators(original)
    prepared = optimizer.optimize(original, num_iterations=2)
    after = _operators(prepared)
    onnx.checker.check_model(prepared, full_check=True)
    onnx.save(prepared, args.output)

    sample = np.random.default_rng(41).random((1, 3, 640, 640), dtype=np.float32)
    if original.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
        sample = sample.astype(np.float16)
    sessions = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for path in (args.input, args.output)
    ]
    outputs = [session.run(None, {"images": sample}) for session in sessions]
    comparisons = []
    for baseline, actual in zip(*outputs, strict=True):
        difference = np.abs(actual.astype(np.float32) - baseline.astype(np.float32))
        comparisons.append(
            {
                "finite": bool(np.isfinite(actual).all()),
                "close_atol1e-4_rtol1e-3": bool(
                    np.allclose(actual, baseline, atol=1e-4, rtol=1e-3)
                ),
                "max_abs": float(difference.max()),
            }
        )
    report = {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "nodes_before": sum(before.values()),
        "nodes_after": sum(after.values()),
        "sequence_operators_before": {
            name: before[name] for name in ("SplitToSequence", "SequenceAt")
        },
        "sequence_operators_after": {
            name: after[name] for name in ("SplitToSequence", "SequenceAt")
        },
        "onnx_checker_full_check": "passed",
        "ort_cpu_graph_equivalence": comparisons,
        "scope": "checkout-only TensorRT preparation; this does not alter public ONNX export",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not all(item["finite"] and item["close_atol1e-4_rtol1e-3"] for item in comparisons):
        message = "Prepared graph failed CPU graph-equivalence tolerance"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
