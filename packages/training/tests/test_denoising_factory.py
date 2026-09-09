# Copyright 2026 Clean RT-DETR contributors
"""Deferred DN construction resolves detector metadata before optimizer creation."""

import pytest
import torch
from test_training_interfaces import ToyDenoisingGenerator, ToyDetector

from clean_rtdetr_training import DenoisingTrainingModel


def test_denoising_factory_registers_parameters_before_first_forward() -> None:
    """An independent detector supplies dimensions and owns initialized DN state."""
    detector = ToyDetector()
    generator = ToyDenoisingGenerator()
    calls: list[tuple[int, int]] = []

    def factory(*, hidden_dim: int, num_classes: int) -> ToyDenoisingGenerator:
        calls.append((hidden_dim, num_classes))
        return generator

    model = DenoisingTrainingModel.from_factory(detector, factory)
    assert calls == [(detector.hidden_dim, detector.num_classes)]
    assert model.model is detector
    assert model.denoising is generator
    torch.testing.assert_close(model.state_dict()["denoising.content"], generator.content)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert any(
        parameter is generator.content
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


@pytest.mark.parametrize("dimension", ["hidden_dim", "num_classes"])
def test_denoising_factory_rejects_incompatible_generator(dimension: str) -> None:
    """Factory composition preserves the direct constructor's validation."""
    detector = ToyDetector()

    def factory(*, hidden_dim: int, num_classes: int) -> ToyDenoisingGenerator:
        generator = ToyDenoisingGenerator()
        generator.hidden_dim = hidden_dim
        generator.num_classes = num_classes
        setattr(generator, dimension, getattr(generator, dimension) + 1)
        return generator

    with pytest.raises(ValueError, match="Denoising embedding must match model"):
        DenoisingTrainingModel.from_factory(detector, factory)
