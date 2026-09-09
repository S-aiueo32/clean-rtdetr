# Copyright 2026 Clean RT-DETR contributors
"""Public DN splitting preserves standard payloads and gradient ownership."""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from clean_rtdetr.contracts import DecoderOutput, DenoisingQueries
from clean_rtdetr_training.contracts import DenoisingBatch, DenoisingMetadata
from clean_rtdetr_training.models.rtdetr import split_denoising_predictions


def _prepared(count: int = 2) -> DenoisingBatch:
    """Prepare a small DN prefix with three matching queries after it."""
    return DenoisingBatch(
        DenoisingQueries(
            torch.randn(2, count, 8),
            torch.randn(2, count, 4),
            torch.zeros(count + 3, count + 3, dtype=torch.bool),
        ),
        DenoisingMetadata((torch.tensor([0]), torch.tensor([0])), 1, count),
    )


def _decoded() -> DecoderOutput:
    """Keep leaf predictions for checking every layer's DN and matching gradients."""
    return DecoderOutput(
        torch.randn(2, 2, 5, 3, requires_grad=True),
        torch.rand(2, 2, 5, 4, requires_grad=True),
    )


def test_split_keeps_metadata_layers_and_both_gradient_routes() -> None:
    """Independent DN and matching losses reach their original query slots."""
    decoded = _decoded()
    prepared = _prepared()
    matching, denoising = split_denoising_predictions(decoded, prepared)
    assert denoising is not None
    assert denoising.metadata is prepared.metadata
    torch.testing.assert_close(matching.logits, decoded.logits[:, :, 2:])
    torch.testing.assert_close(matching.boxes, decoded.boxes[:, :, 2:])
    torch.testing.assert_close(denoising.decoder.logits, decoded.logits[:, :, :2])
    torch.testing.assert_close(denoising.decoder.boxes, decoded.boxes[:, :, :2])
    loss = (
        3 * matching.logits.sum()
        + 5 * matching.boxes.sum()
        + 7 * denoising.decoder.logits.sum()
        + 11 * denoising.decoder.boxes.sum()
    )
    loss.backward()
    expected_logits = torch.full_like(decoded.logits, 3)
    expected_logits[:, :, :2] = 7
    expected_boxes = torch.full_like(decoded.boxes, 5)
    expected_boxes[:, :, :2] = 11
    torch.testing.assert_close(decoded.logits.grad, expected_logits)
    torch.testing.assert_close(decoded.boxes.grad, expected_boxes)


@dataclass(frozen=True)
class ExtendedDecoderOutput(DecoderOutput):
    """Represent a recipe output whose additional state the helper must retain."""

    distribution: Tensor


def test_no_denoising_preserves_custom_output_identity() -> None:
    """The ordinary inference path leaves custom decoder output types untouched."""
    plain = _decoded()
    for decoded in (
        plain,
        ExtendedDecoderOutput(plain.logits, plain.boxes, torch.randn(2, 2, 5, 4, 8)),
    ):
        matching, denoising = split_denoising_predictions(decoded, None)
        assert matching is decoded
        assert denoising is None


def test_denoising_rejects_custom_payload_without_discarding_extra_state() -> None:
    """Recipe-specific tensors require a matching recipe-specific splitting policy."""
    plain = _decoded()
    decoded = ExtendedDecoderOutput(plain.logits, plain.boxes, torch.randn(2, 2, 5, 4, 8))
    with pytest.raises(TypeError, match="custom outputs own their split"):
        split_denoising_predictions(decoded, _prepared())


@pytest.mark.parametrize("count", [0, -1, True, 2.0, 3])
def test_split_rejects_invalid_or_mismatched_prepared_counts(count: int) -> None:
    """Count validation applies to direct helper callers as well as the wrapper."""
    prepared = _prepared()
    object.__setattr__(prepared.metadata, "num_queries", count)
    with pytest.raises(ValueError, match="positive integer matching queries"):
        split_denoising_predictions(_decoded(), prepared)


def test_split_rejects_malformed_prepared_content_and_decoder_layout() -> None:
    """Malformed tensors produce contract errors before query-axis slicing."""
    prepared = _prepared()
    object.__setattr__(prepared.queries, "content", torch.randn(2, 8))
    with pytest.raises(ValueError, match="positive integer matching queries"):
        split_denoising_predictions(_decoded(), prepared)
    decoded = _decoded()
    object.__setattr__(decoded, "boxes", decoded.boxes[0])
    with pytest.raises(ValueError, match=r"shape \[L, B, Q, K/4\]"):
        split_denoising_predictions(decoded, _prepared())


@pytest.mark.parametrize("count", [5, 6])
def test_split_requires_remaining_matching_queries(count: int) -> None:
    """The DN prefix cannot consume or exceed the decoder query axis."""
    with pytest.raises(ValueError, match="must leave matching queries"):
        split_denoising_predictions(_decoded(), _prepared(count))
