# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the fixed-shape tensor-only export boundary."""

from pathlib import Path
from typing import cast

import pytest
import torch
from export_model_fixtures import small_model
from jaxtyping import TypeCheckError

from clean_rtdetr import QuerySelector
from clean_rtdetr.models.rtdetr import BaselineRTDETR
from clean_rtdetr.models.selector import BoxHead
from clean_rtdetr_export import FixedShapeRTDETR, export_onnx
from clean_rtdetr_export.onnx import _comparison

INPUT_SHAPE = (1, 3, 64, 64)
CHECK_COUNT = 4


def model_with_nonzero_box_heads() -> BaselineRTDETR:
    """Build a compact model whose refinement boxes depend on the image."""
    torch.manual_seed(17)
    model = small_model()
    for head in model.decoder.dec_bbox_head:
        assert isinstance(head, BoxHead)
        layer = cast("torch.nn.Linear", head.layers[-1])
        torch.nn.init.normal_(layer.weight, std=0.05)
        torch.nn.init.normal_(layer.bias, std=0.05)
    return model


def test_adapter_copies_weights_and_mixed_training_state() -> None:
    """Specialization must not mutate source parameters or its mixed train/eval state."""
    source = model_with_nonzero_box_heads().train()
    source.encoder.eval()
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    source_parameter_ptrs = {parameter.data_ptr() for parameter in source.parameters()}
    source_training = {name: module.training for name, module in source.named_modules()}

    adapter = FixedShapeRTDETR(source, INPUT_SHAPE)

    assert adapter.training is False
    assert all(module.training is False for module in adapter.modules())
    assert all(module.training == source_training[name] for name, module in source.named_modules())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_state[name])
    assert source_parameter_ptrs.isdisjoint(
        {parameter.data_ptr() for parameter in adapter.model.parameters()}
    )


@torch.inference_mode()
def test_adapter_matches_eager_eval_for_two_inputs() -> None:
    """The tensor-only adapter agrees with the reference model on held-out inputs."""
    source = model_with_nonzero_box_heads().eval()
    adapter = FixedShapeRTDETR(source, INPUT_SHAPE)
    for seed in (31, 32):
        images = torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(seed))
        expected = source(images).final
        actual = adapter(images)
        torch.testing.assert_close(actual[0], expected.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[1], expected.boxes, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((1, 3, 32, 64)),
        torch.zeros((1, 3, 64, 64), dtype=torch.float64),
        torch.zeros((1, 3, 64, 64), dtype=torch.int64),
    ],
)
def test_adapter_rejects_wrong_shape_or_dtype(images: torch.Tensor) -> None:
    """Reject profile violations before running the copied model."""
    adapter = FixedShapeRTDETR(small_model(), INPUT_SHAPE)
    with pytest.raises(
        (ValueError, TypeCheckError), match=r"fixed|float32|type-check|Expected type"
    ):
        adapter(images)


@pytest.mark.parametrize("input_shape", [(1, 3, 63, 64), (2, 3, 64, 64), (1, 1, 64, 64)])
def test_adapter_rejects_invalid_profile_shape(input_shape: tuple[int, int, int, int]) -> None:
    """Require batch-one RGB inputs and dimensions aligned to the backbone stride."""
    with pytest.raises(ValueError, match="fixed"):
        FixedShapeRTDETR(small_model(), input_shape)


def test_adapter_rejects_training_mode() -> None:
    """Export specialization remains eval-only even if explicitly switched to train."""
    adapter = FixedShapeRTDETR(small_model(), INPUT_SHAPE)
    adapter.train()
    with pytest.raises(ValueError, match="eval"):
        adapter(torch.zeros(INPUT_SHAPE))


def test_export_onnx_report_graph_and_runtime(tmp_path: Path) -> None:
    """Export a validated graph and check schema, parity evidence, and ORT input guards."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    destination = tmp_path / "model.onnx"
    model = model_with_nonzero_box_heads().eval()

    report = export_onnx(model, destination, input_shape=INPUT_SHAPE)

    assert report["passed"] is True
    assert report["schema_version"] == 1
    assert report["outputs"] == ["logits", "boxes"]
    assert report["input_shape"] == list(INPUT_SHAPE)
    checks = report["checks"]
    assert isinstance(checks, list)
    assert len(checks) == CHECK_COUNT
    graph = onnx.load(destination)
    assert [value.name for value in graph.graph.input] == ["images"]
    assert [value.name for value in graph.graph.output] == ["logits", "boxes"]
    assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(destination), sess_options=options, providers=["CPUExecutionProvider"]
    )
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
        session.run(None, {"images": torch.zeros((1, 3, 32, 64)).numpy()})
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
        session.run(None, {"images": torch.zeros(INPUT_SHAPE, dtype=torch.int64).numpy()})

    first = torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(81)).numpy()
    second = torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(82)).numpy()
    first_output = session.run(None, {"images": first})
    second_output = session.run(None, {"images": second})
    assert not torch.equal(torch.from_numpy(first_output[0]), torch.from_numpy(second_output[0]))
    assert not torch.equal(torch.from_numpy(first_output[1]), torch.from_numpy(second_output[1]))


def test_export_onnx_refuses_overwrite(tmp_path: Path) -> None:
    """Existing artifacts are never replaced by an export call."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    destination = tmp_path / "existing.onnx"
    destination.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError, match="overwrite"):
        export_onnx(small_model(), destination, input_shape=INPUT_SHAPE)
    assert destination.read_bytes() == b"sentinel"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.1])
def test_parity_gate_rejects_nonfinite_or_inaccurate_output(value: float) -> None:
    """The numerical gate must fail rather than serialize nonfinite error metrics."""
    pytest.importorskip("numpy")
    actual = torch.full((1, 2, 3), value)
    result = _comparison(actual, torch.ones_like(actual))
    assert result["passed"] is False
    if not torch.isfinite(actual).all():
        assert result["max_abs"] is None
        assert result["max_rel"] is None


def test_adapter_rejects_overridden_selection() -> None:
    """A research selector cannot silently lose its overridden forward during lowering."""

    class ResearchSelector(QuerySelector):
        """Represent a variant whose selection semantics need their own lowering."""

    model = small_model()
    model.selector = ResearchSelector(model.selector.config)
    with pytest.raises(ValueError, match="built-in"):
        FixedShapeRTDETR(model, INPUT_SHAPE)
