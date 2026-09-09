# Copyright 2026 Clean RT-DETR contributors
"""Compare bin transforms with the pinned official D-FINE utility functions."""

import ast
from pathlib import Path
from typing import Any

import pytest
import torch

from dfine_recipe.distribution import (
    box_to_distances,
    distance_to_boxes,
    interpolate_bins,
    weighting_function,
)


def _oracle() -> dict[str, Any]:
    """Load only four pure functions, without upstream imports or registries."""
    path = Path(__file__).parent / "reference/dfine_utils.py.txt"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "torch": torch,
        "box_xyxy_to_cxcywh": lambda boxes: torch.cat(
            ((boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]), -1
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reg_max", [4, 8, 32])
def test_projection_interpolation_and_decoding_match_upstream(
    dtype: torch.dtype, reg_max: int
) -> None:
    """Preserve bins, saturated interpolation, box coordinates, and gradients."""
    oracle = _oracle()
    up, scale = 0.5, 4.0
    reference = torch.tensor([[0.3, 0.4, 0.2, 0.1]], dtype=dtype)
    bins = weighting_function(reg_max, up, scale, reference)
    upstream_up = torch.tensor([up], dtype=dtype)
    upstream_scale = torch.tensor([scale], dtype=dtype)
    expected_bins = oracle["weighting_function"](reg_max, upstream_up, upstream_scale)
    torch.testing.assert_close(bins, expected_bins)
    values = torch.tensor([-100.0, -4.0, -0.3, 0.0, 0.7, 4.0, 100.0], dtype=dtype)
    left, right_weight = interpolate_bins(values, bins)
    expected_left, expected_right, _ = oracle["translate_gt"](
        values.float(), reg_max, upstream_scale.float(), upstream_up.float()
    )
    torch.testing.assert_close(left, expected_left.long())
    torch.testing.assert_close(right_weight, expected_right.to(dtype), rtol=2e-5, atol=2e-6)
    distances = torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=dtype, requires_grad=True)
    result = distance_to_boxes(reference, distances, scale)
    expected = oracle["distance2bbox"](reference, distances, scale)
    torch.testing.assert_close(result, expected)
    actual_grad = torch.autograd.grad(result.square().sum(), distances)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), distances)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    xyxy = torch.cat(
        (result[..., :2] - result[..., 2:] / 2, result[..., :2] + result[..., 2:] / 2), -1
    )
    torch.testing.assert_close(box_to_distances(reference, xyxy, scale), distances)


@pytest.mark.parametrize("reg_max", [0, 2, 3, 7])
def test_invalid_bin_count(reg_max: int) -> None:
    """Reject counts that cannot define the nonuniform symmetric projection."""
    with pytest.raises(ValueError, match="even integer"):
        weighting_function(reg_max, 0.5, 4.0, torch.zeros(1))
