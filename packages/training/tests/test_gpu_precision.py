# Copyright 2026 Clean RT-DETR contributors
"""Denoising loss and optimizer integration with autocast feature computation."""

import pytest
import torch
from training_model_fixtures import small_model

from clean_rtdetr_training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    DetectionTarget,
    HungarianMatcher,
    SetCriterion,
)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or not torch.cuda.is_bf16_supported(including_emulation=False),
                reason="Native CUDA BF16 is required",
            ),
        ),
    ],
)
def test_autocast_denoising_updates_parameters(device: str, dtype: torch.dtype) -> None:
    """Full matching, auxiliary and DN losses work without an external dtype adapter."""
    torch.manual_seed(1729)
    model = (
        DenoisingTrainingModel(
            small_model(), DenoisingQueryGenerator(3, 8, DenoisingConfig(num_denoising=4))
        )
        .to(device)
        .train()
    )
    criterion = SetCriterion(matcher=HungarianMatcher())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device == "cuda" and dtype == torch.float16, init_scale=128
    )
    images = torch.rand(2, 3, 64, 64, device=device)
    targets = [
        DetectionTarget(
            torch.tensor([1], device=device),
            torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
        )
        for _ in range(2)
    ]
    initial = model.denoising.class_embed.weight.detach().clone()
    required_steps = 3
    successful_steps = 0
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=dtype):
            output = model(images, targets=targets)
            losses = criterion(output, targets)
            total = torch.stack(list(losses.values())).sum()
        assert "loss_vfl_dn_0" in losses
        assert torch.isfinite(total)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        finite = all(torch.isfinite(g).all() for g in gradients)
        if finite:
            assert any(torch.count_nonzero(g) > 0 for g in gradients)
            successful_steps += 1
        else:
            assert device == "cuda"
            assert dtype == torch.float16
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not finite:
            assert scaler.get_scale() < previous_scale
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if successful_steps == required_steps:
            break
    assert successful_steps == required_steps
    assert not torch.equal(model.denoising.class_embed.weight, initial)
    assert all(torch.isfinite(p).all() for p in model.parameters())
