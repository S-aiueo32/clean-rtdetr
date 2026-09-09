# Copyright 2026 Clean RT-DETR contributors
"""Parity and boundary tests for denoising query construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from denoising_oracle import denoising_oracle

from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.denoising import DenoisingConfig, DenoisingQueryGenerator

TWO_GROUPS = 2
EIGHT_DN_QUERIES = 8


def _targets(dtype: torch.dtype = torch.float32) -> list[DetectionTarget]:
    return [
        DetectionTarget(
            torch.tensor([0, 2]),
            torch.tensor([[0.25, 0.3, 0.2, 0.25], [0.7, 0.65, 0.15, 0.2]], dtype=dtype),
        ),
        DetectionTarget(
            torch.tensor([1]),
            torch.tensor([[0.5, 0.45, 0.3, 0.25]], dtype=dtype),
        ),
    ]


@pytest.mark.parametrize(
    ("config", "seed"),
    [
        (DenoisingConfig(num_denoising=5), 19),
        (DenoisingConfig(num_denoising=1, label_noise_ratio=1.0), 73),
        (DenoisingConfig(num_denoising=7, box_noise_scale=0.0), 91),
    ],
)
def test_values_rng_state_and_embedding_gradients_match_upstream(
    config: DenoisingConfig, seed: int
) -> None:
    """Match upstream values, RNG consumption, metadata, and embedding gradients."""
    generator = DenoisingQueryGenerator(3, 8, config)
    oracle_embed = torch.nn.Embedding(4, 8, padding_idx=3)
    oracle_embed.load_state_dict(generator.class_embed.state_dict())

    torch.manual_seed(seed)
    actual = generator(_targets(), num_queries=4)
    actual_rng = torch.random.get_rng_state()
    torch.manual_seed(seed)
    expected_content, expected_boxes, expected_mask, expected_meta = denoising_oracle(
        _targets(),
        num_classes=3,
        num_queries=4,
        class_embed=oracle_embed,
        num_denoising=config.num_denoising,
        label_noise_ratio=config.label_noise_ratio,
        box_noise_scale=config.box_noise_scale,
    )
    expected_rng = torch.random.get_rng_state()

    assert actual is not None
    assert expected_content is not None
    assert expected_boxes is not None
    assert expected_mask is not None
    assert expected_meta is not None
    torch.testing.assert_close(actual.queries.content, expected_content)
    torch.testing.assert_close(actual.queries.ref_logits, expected_boxes)
    torch.testing.assert_close(actual.queries.attention_mask, expected_mask)
    assert actual.metadata.num_groups == expected_meta["dn_num_group"]
    assert actual.metadata.num_queries == expected_meta["dn_num_split"][0]
    for actual_index, expected_index in zip(
        actual.metadata.positive_indices, expected_meta["dn_positive_idx"], strict=True
    ):
        torch.testing.assert_close(actual_index, expected_index)
    torch.testing.assert_close(actual_rng, expected_rng)

    actual.queries.content.sum().backward()
    expected_content.sum().backward()
    torch.testing.assert_close(generator.class_embed.weight.grad, oracle_embed.weight.grad)
    assert generator.class_embed.weight.grad is not None
    assert torch.equal(generator.class_embed.weight.grad[3], torch.zeros(8))


def test_float64_preserves_dtype_device_and_finite_references() -> None:
    """Follow a generator converted to float64 without silently narrowing boxes."""
    generator = DenoisingQueryGenerator(3, 5).double()
    result = generator(_targets(torch.float64), num_queries=2)
    assert result is not None
    assert result.queries.content.dtype == torch.float64
    assert result.queries.ref_logits.dtype == torch.float64
    assert result.queries.content.device == generator.class_embed.weight.device
    assert torch.isfinite(result.queries.ref_logits).all()


def test_padding_layout_positive_indices_and_attention_groups() -> None:
    """Keep padded positives and cross-group attention in upstream positions."""
    generator = DenoisingQueryGenerator(
        3, 4, DenoisingConfig(num_denoising=5, label_noise_ratio=0.0, box_noise_scale=0.0)
    )
    result = generator(_targets(), num_queries=3)
    assert result is not None
    assert result.metadata.num_groups == TWO_GROUPS
    assert result.metadata.num_queries == EIGHT_DN_QUERIES
    assert [indices.tolist() for indices in result.metadata.positive_indices] == [
        [0, 1, 4, 5],
        [0, 4],
    ]
    mask = result.queries.attention_mask
    assert mask.shape == (11, 11)
    assert mask[8:, :8].all()
    assert mask[:4, 4:8].all()
    assert mask[4:8, :4].all()
    assert not mask[:4, :4].any()
    assert not mask[4:8, 4:8].any()
    assert not mask[:8, 8:].any()
    assert not mask[8:, 8:].any()


def test_disabled_and_all_empty_batches_return_none_without_consuming_rng() -> None:
    """Return no batch and leave RNG untouched when DN has no work."""
    target = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty((0, 4)))
    disabled = DenoisingQueryGenerator(3, 4, DenoisingConfig(num_denoising=0))
    enabled = DenoisingQueryGenerator(3, 4)
    state = torch.random.get_rng_state()
    assert disabled(_targets(), num_queries=2) is None
    assert enabled([target, target], num_queries=2) is None
    torch.testing.assert_close(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_denoising": -1},
        {"label_noise_ratio": -0.1},
        {"label_noise_ratio": 1.1},
        {"label_noise_ratio": float("nan")},
        {"box_noise_scale": -0.1},
        {"box_noise_scale": float("inf")},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float | int]) -> None:
    """Reject negative, nonfinite, and out-of-range noise settings."""
    with pytest.raises(ValueError, match="must"):
        DenoisingConfig(**cast("Any", kwargs))


@pytest.mark.parametrize(("classes", "hidden"), [(0, 4), (3, 0), (True, 4)])
def test_constructor_rejects_invalid_dimensions(classes: int, hidden: int) -> None:
    """Require meaningful integer class and embedding dimensions."""
    with pytest.raises(ValueError, match="positive integer"):
        DenoisingQueryGenerator(classes, hidden)


def test_forward_rejects_empty_batch_query_count_and_invalid_targets() -> None:
    """Validate batch structure and targets before allocating or drawing noise."""
    generator = DenoisingQueryGenerator(3, 4)
    with pytest.raises(ValueError, match="at least one"):
        generator([], num_queries=2)
    with pytest.raises(ValueError, match="positive integer"):
        generator(_targets(), num_queries=0)
    bad_label = replace(_targets()[0], labels=torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="class range"):
        generator([bad_label], num_queries=2)
    with pytest.raises(ValueError, match="share prediction dtype"):
        generator(_targets(torch.float64), num_queries=2)
