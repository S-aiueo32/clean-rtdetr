# Copyright 2026 Clean RT-DETR contributors
"""CUDA AMP regression coverage for the reference training path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from training_model_fixtures import small_model

from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion
from clean_rtdetr_training.models import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

THREE_UPDATES = 3
MAX_ATTEMPTS = 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression")
@pytest.mark.parametrize("init_scale", [65536.0, 128.0])
def test_cuda_amp_scaler_reaches_three_real_updates(  # noqa: PLR0915
    init_scale: float, record_property: Callable[[str, object], None]
) -> None:
    """Initial scaler calibration may skip updates, but must not corrupt parameters."""
    device = torch.device("cuda")
    torch.manual_seed(1729)
    model = small_model()
    wrapped = DenoisingTrainingModel(
        model,
        DenoisingQueryGenerator(
            num_classes=3,
            hidden_dim=8,
            config=DenoisingConfig(num_denoising=4),
        ),
    ).to(device)
    criterion = SetCriterion(matcher=HungarianMatcher()).to(device)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)

    images = torch.randn(2, 3, 64, 64, device=device, dtype=torch.float32)
    targets: Sequence[DetectionTarget] = [
        DetectionTarget(
            torch.tensor([0, 2], device=device, dtype=torch.int64),
            torch.tensor(
                [[0.25, 0.3, 0.2, 0.25], [0.7, 0.65, 0.15, 0.2]],
                device=device,
                dtype=torch.float32,
            ),
        ),
        DetectionTarget(
            torch.tensor([1], device=device, dtype=torch.int64),
            torch.tensor([[0.5, 0.45, 0.3, 0.25]], device=device, dtype=torch.float32),
        ),
    ]

    parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
    assert parameters
    assert all(parameter.dtype is torch.float32 for parameter in parameters)
    actual_updates = 0
    attempts = 0

    # Keep this test self-contained if a CUDA test runner has configured backend flags.
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        while actual_updates < THREE_UPDATES and attempts < MAX_ATTEMPTS:
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = wrapped(images, targets=targets)
            assert all(
                tensor.dtype is torch.float32
                for tensor in (
                    output.predictions.decoder.logits,
                    output.predictions.decoder.boxes,
                    output.predictions.encoder_aux.logits,
                    output.predictions.encoder_aux.boxes,
                )
            )
            # The criterion owns the FP32 public prediction boundary.
            losses = criterion(output, targets)
            loss = torch.stack(list(losses.values())).sum()
            assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            assert gradients
            gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            gradients_nonzero = any(bool((gradient != 0).any()) for gradient in gradients)
            before = [parameter.detach().clone() for parameter in parameters]
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            after = [parameter.detach().clone() for parameter in parameters]

            assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
            if skipped:
                assert not gradients_finite
                assert all(
                    torch.equal(previous, current)
                    for previous, current in zip(before, after, strict=True)
                )
            else:
                assert gradients_finite
                assert gradients_nonzero
                actual_updates += 1
                changed = any(
                    not torch.equal(previous, current)
                    for previous, current in zip(before, after, strict=True)
                )
                assert changed
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn

    record_property("init_scale", init_scale)
    record_property("attempts", attempts)
    record_property("actual_updates", actual_updates)
    record_property("final_scale", scaler.get_scale())
    assert actual_updates == THREE_UPDATES
