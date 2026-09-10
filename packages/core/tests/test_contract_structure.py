# Copyright 2026 Clean RT-DETR contributors
"""Focused regression tests for public and internal tensor contracts."""

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from clean_rtdetr.contracts import (
    DecoderInput,
    EncodedFeatures,
    FeatureSpec,
    MultiScaleFeatures,
)


def _unchecked[T](cls: type[T], **fields: object) -> T:
    """Build a contract object without constructor instrumentation for public checks."""
    value = object.__new__(cls)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    return value


def _features() -> MultiScaleFeatures:
    return MultiScaleFeatures(
        (torch.zeros(2, 4, 4, 4), torch.zeros(2, 6, 2, 2)),
        (FeatureSpec(4, 8), FeatureSpec(6, 16)),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, specs=(FeatureSpec(5, 8), FeatureSpec(6, 16))), "channels"),
        (
            lambda value: replace(value, levels=(value.levels[0], torch.zeros(2, 6, 0, 2))),
            "nonempty",
        ),
        (
            lambda value: replace(value, specs=(FeatureSpec(4, 8), FeatureSpec(6, 8))),
            "strictly increasing",
        ),
        (
            lambda value: replace(
                value, levels=(value.levels[0], value.levels[1].to(dtype=torch.float64))
            ),
            "dtype",
        ),
    ],
)
def test_multiscale_validate_structure_preserves_stage_contracts(
    change: Callable[[MultiScaleFeatures], MultiScaleFeatures], message: str
) -> None:
    """Structure checks retain spec, geometry, and shared placement invariants."""
    with pytest.raises(ValueError, match=message):
        change(_features()).validate_structure()


@pytest.mark.parametrize(
    "bad",
    [
        _unchecked(
            MultiScaleFeatures,
            levels=(torch.zeros(2, 4, 4),),
            specs=(FeatureSpec(4, 8),),
        ),
        _unchecked(
            MultiScaleFeatures,
            levels=(torch.zeros(2, 4, 4, 4, dtype=torch.int64),),
            specs=(FeatureSpec(4, 8),),
        ),
    ],
)
def test_multiscale_validate_rejects_malformed_annotated_tensors(bad: MultiScaleFeatures) -> None:
    """The explicit public validator catches rank and dtype errors itself."""
    with pytest.raises(ValueError, match="Features must"):
        bad.validate()


def _encoded() -> EncodedFeatures:
    shapes = torch.tensor(((2, 2), (1, 6)), dtype=torch.int64)
    sizes = shapes.prod(dim=1)
    return EncodedFeatures(torch.zeros(1, 10, 4), shapes, sizes.cumsum(0) - sizes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda value: replace(
                value, spatial_shapes=torch.tensor(((0, 2), (1, 6)), dtype=torch.int64)
            ),
            "positive",
        ),
        (lambda value: replace(value, memory=torch.zeros(1, 9, 4)), "sequence length"),
        (
            lambda value: replace(value, level_start_index=torch.tensor((0, 5))),
            "prefix sum",
        ),
        (lambda value: replace(value, memory=torch.zeros(1, 0, 4)), "nonempty"),
    ],
)
def test_encoded_validate_structure_preserves_metadata_invariants(
    change: Callable[[EncodedFeatures], EncodedFeatures], message: str
) -> None:
    """Structure checks retain positive areas, sequence length, and offsets."""
    with pytest.raises(ValueError, match=message):
        change(_encoded()).validate_structure()


@pytest.mark.parametrize(
    "bad",
    [
        _unchecked(
            EncodedFeatures,
            memory=torch.zeros(1, 10),
            spatial_shapes=torch.tensor(((2, 2), (1, 6))),
            level_start_index=torch.tensor((0, 4)),
        ),
        _unchecked(
            EncodedFeatures,
            memory=torch.zeros(1, 10, 4),
            spatial_shapes=torch.tensor((2, 2, 1, 6)),
            level_start_index=torch.tensor((0, 4)),
        ),
        _unchecked(
            EncodedFeatures,
            memory=torch.zeros(1, 10, 4),
            spatial_shapes=torch.tensor(((2, 2), (1, 6)), dtype=torch.float32),
            level_start_index=torch.tensor((0, 4)),
        ),
    ],
)
def test_encoded_validate_rejects_malformed_annotated_tensors(bad: EncodedFeatures) -> None:
    """The explicit public validator catches rank and metadata dtype errors."""
    with pytest.raises(ValueError, match=r"Memory|Spatial shapes|Shape metadata"):
        bad.validate()


def _decoder_input() -> DecoderInput:
    return DecoderInput(
        torch.zeros(2, 5, 8),
        torch.zeros(2, 5, 4),
        torch.zeros(5, 5, dtype=torch.bool),
        matching_queries=3,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda value: replace(value, ref_logits=value.ref_logits.requires_grad_()),
            "detached",
        ),
        (lambda value: replace(value, matching_queries=0), "within"),
        (lambda value: replace(value, matching_queries=6), "within"),
        (
            lambda _value: _unchecked(
                DecoderInput,
                content=torch.zeros(2, 0, 8),
                ref_logits=torch.zeros(2, 0, 4),
                attention_mask=torch.zeros(0, 0, dtype=torch.bool),
                matching_queries=3,
            ),
            "nonempty",
        ),
        (
            lambda value: replace(value, ref_logits=value.ref_logits.to(dtype=torch.float64)),
            "dtype",
        ),
    ],
)
def test_decoder_input_validate_structure_preserves_query_contracts(
    change: Callable[[DecoderInput], DecoderInput], message: str
) -> None:
    """Structure checks retain detach, query count, nonempty, and placement rules."""
    with pytest.raises(ValueError, match=message):
        change(_decoder_input()).validate_structure()


@pytest.mark.parametrize(
    "bad",
    [
        _unchecked(
            DecoderInput,
            content=torch.zeros(2, 5),
            ref_logits=torch.zeros(2, 5, 4),
            attention_mask=None,
            matching_queries=3,
        ),
        _unchecked(
            DecoderInput,
            content=torch.zeros(2, 5, 8),
            ref_logits=torch.zeros(2, 5, 3),
            attention_mask=None,
            matching_queries=3,
        ),
        _unchecked(
            DecoderInput,
            content=torch.zeros(2, 5, 8, dtype=torch.int64),
            ref_logits=torch.zeros(2, 5, 4),
            attention_mask=None,
            matching_queries=3,
        ),
    ],
)
def test_decoder_input_validate_rejects_malformed_annotated_tensors(bad: DecoderInput) -> None:
    """The explicit public validator catches query shape and dtype errors."""
    with pytest.raises(ValueError, match=r"Query content|Reference logits"):
        bad.validate()
