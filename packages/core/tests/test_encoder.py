# Copyright 2026 Clean RT-DETR contributors
"""Contract tests for the hybrid feature encoder."""

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from official_oracle import hybrid_encoder_oracle
from torch import Tensor, nn

from clean_rtdetr.contracts import FeatureSpec, MultiScaleFeatures
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig


def _features() -> tuple[MultiScaleFeatures, tuple[FeatureSpec, ...]]:
    """Build the small three-level pyramid used by contract tests."""
    specs = (FeatureSpec(4, 8), FeatureSpec(6, 16), FeatureSpec(8, 32))
    levels = (torch.randn(2, 4, 8, 8), torch.randn(2, 6, 4, 4), torch.randn(2, 8, 2, 2))
    return MultiScaleFeatures(levels, specs), specs


def test_encoder_preserves_levels_and_can_tap_aifi() -> None:
    """Return three projected and fused levels with an optional AIFI tap."""
    features, specs = _features()
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    )
    output = encoder(features, collect_aifi=True)
    assert [tuple(level.shape) for level in output.features.levels] == [
        (2, 8, 8, 8),
        (2, 8, 4, 4),
        (2, 8, 2, 2),
    ]
    assert output.after_aifi is not None
    assert tuple(output.after_aifi.levels[0].shape) == (2, 8, 2, 2)


def test_encoder_tap_retains_gradient() -> None:
    """Preserve the AIFI gradient path to the source feature level."""
    features, specs = _features()
    features = MultiScaleFeatures(tuple(level.requires_grad_() for level in features.levels), specs)
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    )
    output = encoder(features, collect_aifi=True)
    assert output.after_aifi is not None
    output.after_aifi.levels[0].square().mean().backward()
    assert features.levels[2].grad is not None


def test_aifi_parameters_preserve_identity_and_exclude_projection_and_fusion() -> None:
    """Expose the tap's transformer parameters without extra state registration."""
    features, specs = _features()
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    )
    state_keys = tuple(encoder.state_dict())
    parameters = tuple(encoder.aifi_parameters())
    expected = {
        id(parameter)
        for name, parameter in encoder.named_parameters()
        if name.startswith("encoder.")
    }
    assert parameters
    assert {id(parameter) for parameter in parameters} == expected
    assert len(parameters) == len(expected)
    parameters[0].requires_grad = False
    assert tuple(map(id, encoder.aifi_parameters())) == tuple(map(id, parameters))
    assert tuple(encoder.state_dict()) == state_keys
    output = encoder(features, collect_aifi=True)
    assert output.after_aifi is not None
    output.after_aifi.levels[0].square().mean().backward()
    assert parameters[0].grad is None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters
    )
    assert all(
        parameter.grad is None
        for name, parameter in encoder.named_parameters()
        if not name.startswith(("encoder.", "input_proj."))
    )


def test_encoder_rejects_bad_stride_geometry() -> None:
    """Reject a feature pyramid whose spatial heights do not follow strides."""
    features, specs = _features()
    bad = MultiScaleFeatures(
        (features.levels[0], features.levels[1], torch.randn(2, 8, 3, 3)), specs
    )
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    )
    with pytest.raises(ValueError, match="stride ratios"):
        encoder(bad)


def test_encoder_rejects_non_three_level_specs() -> None:
    """Reject encoder construction without exactly three pyramid levels."""
    with pytest.raises(ValueError, match="exactly three"):
        HybridEncoder((FeatureSpec(4, 8), FeatureSpec(6, 16)))


def test_encoder_matches_independent_upstream_oracle_on_rectangular_features() -> None:
    """Compare every fused level after loading identical upstream parameters."""
    specs = (FeatureSpec(4, 8), FeatureSpec(6, 16), FeatureSpec(8, 32))
    config = HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    clean = HybridEncoder(specs, config).eval()
    oracle = hybrid_encoder_oracle(
        in_channels=[4, 6, 8],
        feat_strides=[8, 16, 32],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
    ).eval()
    clean.load_state_dict(oracle.state_dict())
    levels = (torch.randn(2, 4, 8, 12), torch.randn(2, 6, 4, 6), torch.randn(2, 8, 2, 3))
    actual = clean(MultiScaleFeatures(levels, specs), collect_aifi=True)
    expected = oracle(levels)
    for ours, theirs in zip(actual.features.levels, expected, strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("training", [False, True])
def test_encoder_forward_input_and_parameter_gradient_parity(*, training: bool) -> None:
    """Match upstream values and every parameter/input gradient in both modes."""
    specs = (FeatureSpec(4, 8), FeatureSpec(6, 16), FeatureSpec(8, 32))
    config = HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    clean = HybridEncoder(specs, config)
    oracle = hybrid_encoder_oracle(
        in_channels=[4, 6, 8],
        feat_strides=[8, 16, 32],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
    )
    clean.load_state_dict(oracle.state_dict())
    clean.train(training)
    oracle.train(training)
    source = torch.randn(2, 4, 8, 12)
    levels_clean = (
        source.detach().clone().requires_grad_(),
        torch.randn(2, 6, 4, 6, requires_grad=True),
        torch.randn(2, 8, 2, 3, requires_grad=True),
    )
    levels_oracle = tuple(level.detach().clone().requires_grad_() for level in levels_clean)
    ours = clean(MultiScaleFeatures(levels_clean, specs), collect_aifi=True)
    theirs = oracle(levels_oracle)
    for actual, expected in zip(ours.features.levels, theirs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    weights = (1.0, 1.7, 2.3)
    clean_loss = sum(
        weight * level.square().mean()
        for weight, level in zip(weights, ours.features.levels, strict=True)
    )
    oracle_loss = sum(
        weight * level.square().mean() for weight, level in zip(weights, theirs, strict=True)
    )
    clean_loss.backward()
    oracle_loss.backward()
    for (name, actual), (_, expected) in zip(
        clean.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert actual.grad is not None, name
        assert expected.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-5)
    for actual, expected in zip(levels_clean, levels_oracle, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-5)


def test_encoder_aifi_tap_matches_oracle_and_bn_state() -> None:
    """Match the highest-level AIFI tensor and training BatchNorm state."""
    specs = (FeatureSpec(4, 8), FeatureSpec(6, 16), FeatureSpec(8, 32))
    config = HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    clean = HybridEncoder(specs, config).train()
    oracle = hybrid_encoder_oracle(
        in_channels=[4, 6, 8],
        feat_strides=[8, 16, 32],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
    ).train()
    clean.load_state_dict(oracle.state_dict())
    levels = (torch.randn(2, 4, 8, 12), torch.randn(2, 6, 4, 6), torch.randn(2, 8, 2, 3))
    tapped_outputs: list[Tensor] = []
    hook = oracle.get_submodule("encoder.0").register_forward_hook(
        lambda _module, _args, output: tapped_outputs.append(output)
    )
    oracle(levels)
    hook.remove()
    actual = clean(MultiScaleFeatures(levels, specs), collect_aifi=True)
    assert actual.after_aifi is not None
    assert len(tapped_outputs) == 1
    tapped = tapped_outputs[0]
    torch.testing.assert_close(
        actual.after_aifi.levels[0],
        tapped.transpose(1, 2).reshape(2, 8, 2, 3),
        rtol=1e-5,
        atol=1e-5,
    )
    upstream_state = oracle.state_dict()
    for name, value in clean.state_dict().items():
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            torch.testing.assert_close(value, upstream_state[name])


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
    ],
)
def test_encoder_autocast_matches_upstream_aifi(device: str, dtype: torch.dtype) -> None:
    """Keep the AIFI positional embedding FP32 under low-precision autocast."""
    torch.manual_seed(41)
    specs = (FeatureSpec(4, 8), FeatureSpec(6, 16), FeatureSpec(8, 32))
    config = HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    clean = HybridEncoder(specs, config).eval().to(device)
    oracle = (
        hybrid_encoder_oracle(
            in_channels=[4, 6, 8],
            feat_strides=[8, 16, 32],
            hidden_dim=8,
            nhead=2,
            dim_feedforward=16,
            eval_spatial_size=[64, 96],
        )
        .eval()
        .to(device)
    )
    clean.load_state_dict(oracle.state_dict())
    levels = tuple(
        torch.randn(2, channels, height, width, device=device)
        for channels, height, width in ((4, 8, 12), (6, 4, 6), (8, 2, 3))
    )
    with torch.autocast(device_type=device, dtype=dtype):
        actual = clean(MultiScaleFeatures(levels, specs), collect_aifi=True)
        expected = oracle(levels)
    for ours, theirs in zip(actual.features.levels, expected, strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_encoder_tap_does_not_change_eval_predictions() -> None:
    """Feature collection is observational and leaves the forward values unchanged."""
    features, specs = _features()
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    ).eval()
    without = encoder(features).features
    with_tap = encoder(features, collect_aifi=True).features
    for ours, theirs in zip(without.levels, with_tap.levels, strict=True):
        torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"hidden_dim": 10, "num_heads": 3},
        {"dropout": 1.0},
        {"pe_temperature": float("inf")},
        {"depth_mult": 0.0},
    ],
)
def test_encoder_rejects_invalid_config(kwargs: dict[str, object]) -> None:
    """Reject invalid dimensions, probabilities, and fusion settings."""
    with pytest.raises(ValueError, match=r"Encoder|hidden|dropout|Position|Fusion"):
        replace(HybridEncoderConfig(), **kwargs)


def test_encoder_rejects_bad_width_geometry() -> None:
    """Reject widths that violate the declared stride ratios."""
    features, specs = _features()
    bad = MultiScaleFeatures(
        (features.levels[0], features.levels[1], torch.randn(2, 8, 2, 3)), specs
    )
    encoder = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    )
    with pytest.raises(ValueError, match="widths"):
        encoder(bad)


def test_v2_encoder_uses_official_named_projection_keys() -> None:
    """Injected projections expose v2 names while retaining anonymous v1 keys."""
    _, specs = _features()
    v1_keys = HybridEncoder(
        specs, HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16)
    ).state_dict()
    projections = [
        nn.Sequential(
            OrderedDict(
                (
                    ("conv", nn.Conv2d(spec.channels, 8, 1, bias=False)),
                    ("norm", nn.BatchNorm2d(8)),
                )
            )
        )
        for spec in specs
    ]
    v2_keys = HybridEncoder(
        specs,
        HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
        input_proj=projections,
    ).state_dict()
    assert "input_proj.0.0.weight" in v1_keys
    assert "input_proj.0.conv.weight" in v2_keys
    assert "input_proj.0.norm.weight" in v2_keys
