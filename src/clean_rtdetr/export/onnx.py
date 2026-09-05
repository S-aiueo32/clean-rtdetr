# Copyright 2026 Clean RT-DETR contributors
"""Export and verify fixed-shape FP32 inference using optional ONNX packages."""

import copy
import hashlib
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import torch
from jaxtyping import Float32, Shaped
from torch import Tensor

from clean_rtdetr.export.adapter import DEFAULT_SHAPE, FixedShapeRTDETR
from clean_rtdetr.models.decoder import DecoderPlan
from clean_rtdetr.models.rtdetr import RTDETR

ATOL = 1e-5
RTOL = 1e-4
OPSET = 18
SEEDS = (41, 42)
OUTPUT_NAMES = ("logits", "boxes")


class ExportVerificationError(RuntimeError):
    """Expose a failed numerical report so callers can persist the evidence."""

    def __init__(self, report: dict[str, object]) -> None:
        """Retain the same report schema as a successful export."""
        super().__init__("ONNX output parity failed")
        self.report = report


def _tensor_hash(tensor: Shaped[Tensor, "..."]) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _comparison(
    actual: Float32[Tensor, "batch queries channels"],
    expected: Float32[Tensor, "batch queries channels"],
) -> dict[str, object]:
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "passed": finite and torch.allclose(actual, expected, atol=ATOL, rtol=RTOL),
        "shape": list(actual.shape),
        "max_abs": float(difference.max()) if finite else None,
        "max_rel": float((difference / expected.abs().clamp_min(1e-12)).max()) if finite else None,
        "actual_sha256": _tensor_hash(actual),
        "expected_sha256": _tensor_hash(expected),
    }


def export_onnx(
    model: RTDETR,
    destination: Path,
    *,
    input_shape: tuple[int, int, int, int] = DEFAULT_SHAPE,
) -> dict[str, object]:
    """Write opset-18 ONNX and verify both held-out seeds against CPU FP32 eager.

    Install the ``export`` extra first. The model and its train/eval state remain
    unchanged. The returned JSON-compatible report describes numerical parity,
    not dataset accuracy or latency. Existing artifact paths are rejected.
    """
    # Runtime-loaded plugins keep the base development/type-check environment
    # independent of ONNX; dependency versions are pinned by the export extra.
    try:
        onnx = import_module("onnx")
        ort = import_module("onnxruntime")
        import_module("onnxscript")
    except ImportError as error:
        msg = "ONNX export requires the clean-rtdetr[export] optional dependencies"
        raise ImportError(msg) from error

    destination = Path(destination)
    if destination.exists():
        msg = f"Refusing to overwrite existing ONNX artifact: {destination}"
        raise FileExistsError(msg)
    reference = copy.deepcopy(model).cpu().float().eval()
    adapter = FixedShapeRTDETR(reference, input_shape)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Export input is distinct from both verification inputs.
    sample = torch.rand(input_shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        program = torch.onnx.export(
            adapter,
            (sample,),
            dynamo=True,
            opset_version=OPSET,
            input_names=["images"],
            output_names=list(OUTPUT_NAMES),
            external_data=False,
            optimize=False,
        )
    if program is None:
        msg = "The dynamo exporter did not return an ONNX program"
        raise RuntimeError(msg)
    program.save(destination, external_data=False)
    graph = onnx.load(destination)
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(destination), sess_options=options, providers=["CPUExecutionProvider"]
    )
    expected_outputs = [
        [input_shape[0], model.selector.config.num_queries, model.decoder.config.num_classes],
        [input_shape[0], model.selector.config.num_queries, 4],
    ]
    if [(value.name, value.shape, value.type) for value in session.get_inputs()] != [
        ("images", list(input_shape), "tensor(float)")
    ] or [(value.name, value.shape, value.type) for value in session.get_outputs()] != [
        (name, shape, "tensor(float)")
        for name, shape in zip(OUTPUT_NAMES, expected_outputs, strict=True)
    ]:
        msg = "Exported graph does not match the fixed schema-v1 input/output contract"
        raise RuntimeError(msg)
    checks = []
    passed = True
    with torch.no_grad():
        for seed in SEEDS:
            images = torch.rand(input_shape, generator=torch.Generator().manual_seed(seed))
            expected = reference(images, plan=DecoderPlan(collect_intermediate=False)).final
            lowered = adapter(images)
            actual = session.run(list(OUTPUT_NAMES), {"images": images.numpy()})
            for name, eager, adapted, runtime in zip(
                OUTPUT_NAMES, (expected.logits, expected.boxes), lowered, actual, strict=True
            ):
                adapter_check = _comparison(adapted, eager)
                runtime_check = _comparison(torch.from_numpy(runtime), eager)
                passed = passed and bool(adapter_check["passed"]) and bool(runtime_check["passed"])
                checks.append(
                    {
                        "seed": seed,
                        "input_sha256": _tensor_hash(images),
                        "name": name,
                        "adapter": adapter_check,
                        "runtime": runtime_check,
                    }
                )
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": passed,
        "artifact": str(destination),
        "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "opset": OPSET,
        "input_shape": list(input_shape),
        "input_dtype": "float32",
        "outputs": list(OUTPUT_NAMES),
        "box_format": "normalized cxcywh",
        "decoder_layers": model.decoder.config.num_layers,
        "provider": "CPUExecutionProvider",
        "export_optimization": False,
        "runtime_optimization": "ORT_DISABLE_ALL",
        "torch_num_threads": torch.get_num_threads(),
        "versions": {
            name: version(name) for name in ("torch", "onnx", "onnxscript", "onnxruntime")
        },
        "atol": ATOL,
        "rtol": RTOL,
        "checks": checks,
        "validation_scope": "Seeded synthetic CPU FP32 output parity; AP and latency unmeasured",
    }
    if not passed:
        raise ExportVerificationError(report)
    return report
