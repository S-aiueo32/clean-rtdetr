# Copyright 2026 Clean RT-DETR contributors
# ruff: noqa: C901, PLC0415, TRY003, EM101, EM102, PLR0913, PLR0917
"""Bounded CUDA deployment validation for the official R50vd checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import statistics
import time
import traceback
from collections import Counter
from collections.abc import Callable  # noqa: TC003 - readable timing callback contract
from pathlib import Path
from typing import Any, Protocol, cast

import torch

CHECKPOINT_SHA256 = "dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45"
SHAPE = (1, 3, 640, 640)


class _Prediction(Protocol):
    logits: torch.Tensor
    boxes: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "latency_ms": values,
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "mean_ms": statistics.mean(values),
        "images_per_second": 1000 / statistics.mean(values),
    }


def _timed(
    forward: Callable[[], object], warmup: int, iterations: int, *, wall: bool
) -> dict[str, Any]:
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(iterations):
        if wall:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            forward()
            torch.cuda.synchronize()
            values.append((time.perf_counter_ns() - start) / 1e6)
        else:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {
        "clock": "wall_with_cuda_sync" if wall else "cuda_events",
        "warmup": warmup,
        "iterations": iterations,
        **_summary(values),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual, expected = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual).all())
    difference = (actual - expected).abs()
    return {
        "shape": list(actual.shape),
        "finite": finite,
        "ordered_rows_close": finite
        and bool(torch.allclose(actual, expected, atol=1e-4, rtol=1e-3)),
        "max_abs": float(difference.max()) if bool(torch.isfinite(difference).all()) else None,
    }


def _profile(path: str) -> dict[str, Any]:
    events = json.loads(Path(path).read_text())
    providers = Counter(
        str(event.get("args", {}).get("provider"))
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return {
        "node_kernel_events_by_provider": dict(sorted(providers.items())),
        "cpu_fallback_detected": providers["CPUExecutionProvider"] > 0,
    }


def _ort(
    path: Path,
    images: torch.Tensor,
    eager: _Prediction,
    provider: str,
    *,
    reference_precision: str = "fp32",
    fp32_reference: _Prediction | None = None,
) -> dict[str, Any]:
    try:
        np = importlib.import_module("numpy")
        ort = importlib.import_module("onnxruntime")
    except ImportError as error:
        return {"status": "unavailable", "reason": f"missing dependency: {error.name}"}
    available = ort.get_available_providers()
    if provider not in available:
        return {
            "status": "unavailable",
            "reason": f"{provider} unavailable",
            "available": available,
            "onnxruntime": ort.__version__,
        }
    options = ort.SessionOptions()
    options.enable_profiling = True
    try:
        provider_options = (
            {"trt_fp16_enable": "True"}
            if (provider == "TensorrtExecutionProvider" and "fp16" in path.stem.lower())
            else {}
        )
        requested = [(provider, provider_options), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), sess_options=options, providers=requested)
        input_type = session.get_inputs()[0].type
        input_dtype = np.float16 if input_type == "tensor(float16)" else np.float32
        outputs = session.run(
            None, {"images": images.cpu().numpy().astype(input_dtype, copy=False)}
        )
        profile_path = session.end_profiling()
        checks = {
            name: _compare(torch.from_numpy(value), expected)
            for name, value, expected in zip(
                ("logits", "boxes"), outputs, (eager.logits.cpu(), eager.boxes.cpu()), strict=True
            )
        }
        fp32_diagnostic = None
        if fp32_reference is not None:
            fp32_diagnostic = {
                name: _compare(torch.from_numpy(value), expected)
                for name, value, expected in zip(
                    ("logits", "boxes"),
                    outputs,
                    (fp32_reference.logits.cpu(), fp32_reference.boxes.cpu()),
                    strict=True,
                )
            }
        finite = all(value["finite"] for value in checks.values())
        profile = _profile(profile_path)
        requested_events = profile["node_kernel_events_by_provider"].get(provider, 0)
        if not finite:
            status = "failed_nonfinite_output"
        elif requested_events == 0:
            status = "provider_not_executed"
        else:
            status = "executed_requested_provider"
        return {
            "status": status,
            "requested_provider": provider,
            "requested_provider_options": provider_options,
            "onnxruntime": ort.__version__,
            "active_providers": session.get_providers(),
            "input_type": input_type,
            "checks": checks,
            "reference_precision": reference_precision,
            "fp32_reference_diagnostic": fp32_diagnostic,
            "query_identity_alignment": {
                "status": "unverified",
                "reason": "fixed-shape export does not expose selector source_indices",
            },
            "profile": profile,
            "profiling_artifact": profile_path,
        }
    except Exception as error:  # noqa: BLE001 - preserve optional provider build failure
        return {
            "status": "failed",
            "onnxruntime": ort.__version__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def _case(
    model: torch.nn.Module,
    images: torch.Tensor,
    source: torch.Tensor,
    reference: _Prediction,
    dtype: torch.dtype | None,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    from clean_rtdetr.inference import postprocess, preprocess_images

    def forward() -> object:
        with torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext():
            return model(images).final

    def end_to_end() -> object:
        prepared = preprocess_images([source])
        with torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext():
            prediction = model(prepared.images.cuda().to(dtype=images.dtype)).final
        return postprocess(prediction, prepared.original_sizes_wh.cuda())

    model_only = _timed(forward, warmup, iterations, wall=False)
    torch.cuda.empty_cache()
    full = _timed(end_to_end, warmup, iterations, wall=True)
    output = cast("_Prediction", forward())
    finite = bool(torch.isfinite(output.logits).all() and torch.isfinite(output.boxes).all())
    return {
        "status": "passed_execution" if finite else "failed_nonfinite_output",
        "model_only": model_only,
        "end_to_end": full,
        "fp32_reference_comparison": {
            "logits": _compare(output.logits.cpu(), reference.logits.cpu()),
            "boxes": _compare(output.boxes.cpu(), reference.boxes.cpu()),
        },
    }


def run(
    checkpoint: Path,
    output: Path,
    onnx: Path | None,
    onnx_fp16: Path | None,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Run official-weight CUDA checks and write their JSON report."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deployment validation")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    from clean_rtdetr import rtdetr_r50vd
    from tools.compatibility import load_official_model_weights

    digest = _sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = rtdetr_r50vd().eval()
    load_official_model_weights(checkpoint, model=model, source="ema", expected_sha256=digest)
    images = torch.randn(SHAPE).cuda()
    source = torch.randint(0, 256, (3, 713, 997), dtype=torch.uint8)
    model.cuda()
    report: dict[str, Any] = {
        "schema_version": 2,
        "shape": list(SHAPE),
        "seed": 41,
        "checkpoint_sha256": digest,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32": False,
        "scope": (
            "official EMA; end-to-end includes public preprocessing, transfer, model, "
            "and public postprocessing"
        ),
        "results": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    persist()
    with torch.inference_mode():
        reference = model(images).final
        for name, dtype in (("fp32", None), ("amp_fp16", torch.float16)):
            torch.cuda.empty_cache()
            report["results"][name] = _case(
                model, images, source, reference, dtype, warmup, iterations
            )
            persist()
        native_reference: _Prediction | None = None
        torch.cuda.empty_cache()
        model.half()
        try:
            report["results"]["native_fp16"] = _case(
                model, images.half(), source, reference, None, warmup, iterations
            )
            native_reference = model(images.half()).final
        except Exception as error:  # noqa: BLE001 - retain experimental failure and continue
            report["results"]["native_fp16"] = {
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        persist()
        for path, suffix in ((onnx, "fp32"), (onnx_fp16, "fp16")):
            if path is not None:
                graph_reference = (
                    native_reference
                    if suffix == "fp16" and native_reference is not None
                    else reference
                )
                report[f"onnx_cuda_{suffix}"] = _ort(
                    path,
                    images,
                    graph_reference,
                    "CUDAExecutionProvider",
                    reference_precision="native_fp16" if suffix == "fp16" else "fp32",
                    fp32_reference=reference if suffix == "fp16" else None,
                )
                persist()
                report[f"onnx_tensorrt_{suffix}"] = _ort(
                    path,
                    images,
                    graph_reference,
                    "TensorrtExecutionProvider",
                    reference_precision="native_fp16" if suffix == "fp16" else "fp32",
                    fp32_reference=reference if suffix == "fp16" else None,
                )
                persist()
    return report


def main() -> None:
    """Parse command-line arguments and execute the probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--onnx-fp16", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.checkpoint,
                args.output,
                args.onnx,
                args.onnx_fp16,
                args.warmup,
                args.iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
