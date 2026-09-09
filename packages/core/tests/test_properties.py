# Copyright 2026 Clean RT-DETR contributors
"""Property-based tests for tensor geometry and metadata contracts."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from clean_rtdetr.contracts import EncodedFeatures
from clean_rtdetr.geometry import generate_anchors

positive_dimension = st.integers(min_value=1, max_value=32)
spatial_shape = st.tuples(positive_dimension, positive_dimension)
spatial_shapes = st.lists(spatial_shape, min_size=1, max_size=4).map(tuple)


@settings(max_examples=40, deadline=None)
@given(shapes=spatial_shapes)
def test_anchor_generation_preserves_sequence_and_validity_invariants(
    shapes: tuple[tuple[int, int], ...],
) -> None:
    """Every generated level contributes exactly its spatial area."""
    anchors, valid = generate_anchors(shapes)
    sequence = sum(height * width for height, width in shapes)

    assert anchors.shape == (1, sequence, 4)
    assert valid.shape == (1, sequence, 1)
    assert torch.isfinite(anchors[valid.expand_as(anchors)]).all()
    assert torch.isposinf(anchors[~valid.expand_as(anchors)]).all()


@settings(max_examples=40, deadline=None)
@given(
    shapes=spatial_shapes,
    batch=st.integers(min_value=1, max_value=3),
    channels=st.integers(min_value=1, max_value=16),
)
def test_encoded_features_accepts_exact_prefix_sum_metadata(
    shapes: tuple[tuple[int, int], ...], batch: int, channels: int
) -> None:
    """Valid metadata remains accepted for arbitrary small multi-level layouts."""
    sizes = torch.tensor(shapes, dtype=torch.int64).prod(dim=1)
    memory = torch.zeros(batch, int(sizes.sum()), channels)
    offsets = sizes.cumsum(0) - sizes

    EncodedFeatures(
        memory,
        torch.tensor(shapes, dtype=torch.int64),
        offsets,
    ).validate()


@pytest.mark.parametrize("bad_eps", [0.0, -0.1, 0.5, 1.0])
def test_anchor_generation_rejects_invalid_epsilon(bad_eps: float) -> None:
    """The probability margin must leave room on both sides of the interval."""
    with pytest.raises(ValueError, match="eps"):
        generate_anchors(((2, 2),), eps=bad_eps)
