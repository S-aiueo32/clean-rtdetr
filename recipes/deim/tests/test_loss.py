# Copyright 2026 Clean RT-DETR contributors
"""Equation checks for the caller-owned MAL implementation."""

import math

import pytest
import torch

from deim_recipe.loss import MatchabilityAwareLoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mal_values_and_analytic_gradients(dtype: torch.dtype) -> None:
    """Match zero/partial/full IoU and negatives without differentiating weights."""
    logits = torch.tensor([[[-2.0, 0.0, 2.0, -1.0]]], dtype=dtype, requires_grad=True)
    scores = torch.tensor([[[0.0, 0.5, 1.0, 0.0]]], dtype=dtype, requires_grad=True)
    positive = torch.tensor([[[True, True, True, False]]])
    gamma, normalizer = 1.5, 3
    loss = MatchabilityAwareLoss(gamma)(logits, scores, positive, normalizer)
    expected = 0.0
    gradients = []
    for value, iou, matched in zip(
        logits.detach().flatten().tolist(),
        scores.detach().flatten().tolist(),
        positive.flatten().tolist(),
        strict=True,
    ):
        probability = 1 / (1 + math.exp(-value))
        weight = 1.0 if matched else probability**gamma
        target = iou**gamma
        expected += weight * (math.log1p(math.exp(value)) - target * value) / normalizer
        gradients.append(weight * (probability - target) / normalizer)
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=dtype))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[gradients]], dtype=dtype))
    assert scores.grad is None


@pytest.mark.parametrize("gamma", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_gamma(gamma: float) -> None:
    """Reject invalid soft-target exponents."""
    with pytest.raises(ValueError, match="gamma"):
        MatchabilityAwareLoss(gamma)
