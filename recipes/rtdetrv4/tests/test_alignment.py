# Copyright 2026 Clean RT-DETR contributors
"""Verify cosine alignment, resizing and student-only derivatives."""

import pytest
import torch
from torch.autograd import gradcheck

from rtdetrv4_recipe.criterion import alignment_loss


def test_value_and_analytic_gradient() -> None:
    """A 3-4-5 vector has a closed-form cosine value and derivative."""
    student = torch.tensor([3.0, 4.0], dtype=torch.float64).reshape(1, 2, 1, 1)
    student.requires_grad_()
    teacher = torch.tensor([1.0, 0.0], dtype=torch.float64).reshape(1, 2, 1, 1)
    teacher.requires_grad_()
    loss = alignment_loss(student, teacher)
    torch.testing.assert_close(loss, torch.tensor(0.4, dtype=torch.float64))
    loss.backward()
    expected = torch.tensor([-16 / 125, 12 / 125], dtype=torch.float64).reshape_as(student)
    torch.testing.assert_close(student.grad, expected)
    assert teacher.grad is None


def test_teacher_bilinear_resize_and_student_gradcheck() -> None:
    """Four teacher patches interpolate to their mean at the single student site."""
    teacher = torch.tensor([[[[0.0, 2.0], [0.0, 2.0]], [[0.0, 0.0], [2.0, 2.0]]]])
    student = torch.tensor([1.0, 1.0]).reshape(1, 2, 1, 1)
    torch.testing.assert_close(alignment_loss(student, teacher), torch.tensor(0.0))
    torch.manual_seed(41)
    source = torch.randn(2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 4, 3, 2, dtype=torch.float64)
    assert gradcheck(lambda value: alignment_loss(value, target), (source,))


@pytest.mark.parametrize("scale", [0.0, 1e-30, 1.0])
def test_zero_and_tiny_features_remain_finite(scale: float) -> None:
    """Exercise the normalization epsilons used by the upstream cosine path."""
    student = torch.full((2, 3, 2, 2), scale, requires_grad=True)
    loss = alignment_loss(student, torch.zeros_like(student))
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize(
    "teacher",
    [torch.zeros(1, 4, 2, 2), torch.zeros(2, 3, 2, 2), torch.zeros(1, 3, 0, 2)],
)
def test_incompatible_maps_are_rejected(teacher: torch.Tensor) -> None:
    """Reject dimension mismatches before broadcasting can hide them."""
    with pytest.raises(ValueError, match="must share"):
        alignment_loss(torch.zeros(1, 3, 2, 2), teacher)
