# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the PResNet backbone contract."""

from typing import cast

import pytest
import torch
from official_oracle import backbone_oracle

from clean_rtdetr.contracts import FeatureSpec
from clean_rtdetr.models.backbone import FrozenBatchNorm2d, PResNet


def test_r50vd_shapes_specs_and_gradients() -> None:
    """Return the expected three feature levels and propagate gradients."""
    model = PResNet(freeze_at=-1, freeze_norm=False)
    image = torch.randn(2, 3, 64, 96, requires_grad=True)
    output = model(image)
    assert output.specs == (FeatureSpec(512, 8), FeatureSpec(1024, 16), FeatureSpec(2048, 32))
    assert [tuple(level.shape) for level in output.levels] == [
        (2, 512, 8, 12),
        (2, 1024, 4, 6),
        (2, 2048, 2, 3),
    ]
    sum(level.mean() for level in output.levels).backward()
    assert image.grad is not None
    assert next(model.conv1.parameters()).grad is not None


@pytest.mark.parametrize(
    ("depth", "channels"),
    [
        (18, (128, 256, 512)),
        (34, (128, 256, 512)),
        (50, (512, 1024, 2048)),
        (101, (512, 1024, 2048)),
    ],
)
def test_depth_architecture_and_specs(depth: int, channels: tuple[int, ...]) -> None:
    """Select the official block type, stage counts, and output channels."""
    model = PResNet(depth=depth, freeze_at=-1, freeze_norm=False)
    assert model.specs == tuple(
        FeatureSpec(channel, stride) for channel, stride in zip(channels, (8, 16, 32), strict=True)
    )
    assert model.checkpoint_config()["depth"] == depth
    upstream = backbone_oracle(
        depth=depth,
        variant="d",
        num_stages=4,
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    assert model.state_dict().keys() == upstream.state_dict().keys()


def test_default_depth_remains_r50vd() -> None:
    """Keep construction without a depth argument compatible with R50vd."""
    model = PResNet()
    assert model.checkpoint_config()["depth"] == 50  # noqa: PLR2004 - documented default depth
    assert model.specs == (FeatureSpec(512, 8), FeatureSpec(1024, 16), FeatureSpec(2048, 32))


@pytest.mark.parametrize("depth", [17, 152, True])
def test_depth_validation(depth: object) -> None:
    """Reject depths outside the official supported configurations."""
    with pytest.raises(ValueError, match="depth"):
        PResNet(depth=cast("int", depth))


def test_forward_and_gradient_parity_with_upstream() -> None:
    """Match upstream outputs and input gradients after transferring its weights."""
    torch.manual_seed(11)
    upstream = backbone_oracle(
        depth=50,
        variant="d",
        num_stages=4,
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    clean = PResNet(freeze_at=-1, freeze_norm=False)
    clean.load_state_dict(upstream.state_dict())
    upstream.train()
    clean.train()
    upstream_input = torch.randn(2, 3, 64, 96, requires_grad=True)
    clean_input = upstream_input.detach().clone().requires_grad_()
    upstream_outputs = upstream(upstream_input)
    clean_outputs = clean(clean_input).levels
    for expected, actual in zip(upstream_outputs, clean_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(output.mean() for output in upstream_outputs).backward()
    sum(output.mean() for output in clean_outputs).backward()
    torch.testing.assert_close(clean_input.grad, upstream_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        next(clean.parameters()).grad, next(upstream.parameters()).grad, rtol=0, atol=0
    )


def test_training_batch_norm_state_parity() -> None:
    """Update trainable BatchNorm statistics exactly as upstream does."""
    upstream = backbone_oracle(
        depth=50,
        variant="d",
        num_stages=4,
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    clean = PResNet(freeze_at=-1, freeze_norm=False)
    clean.load_state_dict(upstream.state_dict())
    upstream.train()
    clean.train()
    image = torch.randn(2, 3, 64, 96)
    upstream(image)
    clean(image)
    for clean_module, upstream_module in zip(clean.modules(), upstream.modules(), strict=True):
        if isinstance(clean_module, torch.nn.BatchNorm2d):
            torch.testing.assert_close(clean_module.running_mean, upstream_module.running_mean)
            torch.testing.assert_close(clean_module.running_var, upstream_module.running_var)


def test_freezing_matches_upstream_semantics() -> None:
    """Freeze the stem and first two stages, and replace every BatchNorm."""
    model = PResNet(freeze_at=2, freeze_norm=True)
    assert all(not parameter.requires_grad for parameter in model.conv1.parameters())
    assert all(not parameter.requires_grad for parameter in model.res_layers[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.res_layers[1].parameters())
    assert any(parameter.requires_grad for parameter in model.res_layers[2].parameters())
    assert all(
        isinstance(module, FrozenBatchNorm2d)
        for module in model.modules()
        if module.__class__.__name__ == "FrozenBatchNorm2d"
    )
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())


@pytest.mark.parametrize("freeze_at", [-2, 4, True])
def test_freeze_at_validation(freeze_at: object) -> None:
    """Reject freeze levels outside the upstream supported range."""
    with pytest.raises(ValueError, match="freeze_at"):
        PResNet(freeze_at=cast("int", freeze_at))


@pytest.mark.parametrize("shape", [(1, 3, 63, 64), (1, 1, 64, 64), (0, 3, 64, 64)])
def test_input_validation(shape: tuple[int, ...]) -> None:
    """Reject malformed image batches before convolution."""
    with pytest.raises(ValueError, match="Input"):
        PResNet(freeze_norm=False)(torch.randn(shape))
