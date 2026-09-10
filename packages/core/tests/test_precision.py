# Copyright 2026 Clean RT-DETR contributors
"""Low-precision computation retains FP32 prediction and coordinate contracts."""

import pytest
import torch
from core_model_fixtures import small_model

from clean_rtdetr.models.attention import deformable_attention_core

CUDA = pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_half_model_retains_valid_prediction_contract(device: str) -> None:
    """Half weights and images traverse every stage without mixing linear dtypes."""
    torch.manual_seed(1729)
    model = small_model().to(device).half().eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64, device=device).half())
    output.decoder.validate()
    for tensor in (output.final.logits, output.final.boxes, output.encoder_aux.logits):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_autocast_prediction_contract(device: str) -> None:
    """Autocast features must not leak mixed logits/box dtypes into the public output."""
    torch.manual_seed(1729)
    model = small_model().to(device).eval()
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        output = model(torch.rand(1, 3, 64, 64, device=device))
    output.decoder.validate()
    assert output.final.logits.dtype == output.final.boxes.dtype == torch.float32
    assert output.encoder_aux.logits.dtype == output.encoder_aux.boxes.dtype == torch.float32
    assert torch.isfinite(output.final.logits).all()
    assert torch.isfinite(output.final.boxes).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_sampling_retains_coordinate_gradients(dtype: torch.dtype) -> None:
    """Interpolation handles FP32 locations and half values, preserving both gradients."""
    value = torch.arange(4, dtype=dtype).reshape(1, 4, 1, 1).requires_grad_()
    locations = torch.tensor([[[[[[0.4, 0.6]]]]]], requires_grad=True)
    weights = torch.ones(1, 1, 1, 1, 1, dtype=dtype)
    shapes = torch.tensor([[2, 2]])
    expected = deformable_attention_core(value.float(), shapes, locations, weights.float())
    actual = deformable_attention_core(value, shapes, locations, weights)
    torch.testing.assert_close(actual, expected.to(dtype))
    actual.sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert locations.grad is not None
    assert torch.isfinite(locations.grad).all()
    assert torch.count_nonzero(locations.grad) > 0
