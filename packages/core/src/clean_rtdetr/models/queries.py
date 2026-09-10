# Copyright 2026 Clean RT-DETR contributors
"""Assemble decoder query tensors without supervision metadata."""

import torch

from clean_rtdetr.contracts import (
    BOX_COORDINATES,
    QUERY_RANK,
    DecoderInput,
    DenoisingQueries,
    QuerySet,
)
from clean_rtdetr.models.interfaces import Assembler


class QueryAssembler(Assembler):
    """Concatenate optional DN queries and own the initial reference detach boundary."""

    def forward(
        self,
        matching: QuerySet,
        denoising: DenoisingQueries | None = None,
    ) -> DecoderInput:
        """Place DN queries first, matching upstream ordering, then detach all references.

        With D denoising slots and Q matching slots, the decoder layout is::

            query axis:  [ DN: 0 .. D-1 | matching: D .. D+Q-1 ]
            content:     concatenate without detaching
            ref_logits:  concatenate -> detach -> decoder
            mask:        supplied [D+Q, D+Q], True blocks attention

        Without DN, D is zero and no attention mask is supplied. Detaching here
        cuts the initial box-reference gradient for both branches while leaving
        DN embedding gradients and the selector's auxiliary box gradients intact.
        ``matching_queries`` records Q so downstream consumers can locate the tail.
        """
        matching_queries = matching.content.shape[1]

        content = matching.content
        ref_logits = matching.ref_logits
        attention_mask = None
        if denoising is not None:
            self._validate_denoising(matching, denoising)
            content = torch.cat((denoising.content, content), dim=1)
            ref_logits = torch.cat((denoising.ref_logits, ref_logits), dim=1)
            attention_mask = denoising.attention_mask

        assembled = DecoderInput(
            content=content,
            ref_logits=ref_logits.detach(),
            attention_mask=attention_mask,
            matching_queries=matching_queries,
        )
        assembled.validate_structure()
        return assembled

    def _validate_denoising(self, matching: QuerySet, denoising: DenoisingQueries) -> None:
        """Check DN dimensions and mask coverage before concatenating queries."""
        if denoising.content.ndim != QUERY_RANK or denoising.ref_logits.ndim != QUERY_RANK:
            msg = "Denoising queries must have rank-three content and reference tensors"
            raise ValueError(msg)
        batch, matching_queries, channels = matching.content.shape
        denoising_queries = denoising.content.shape[1]
        if denoising.content.shape[:1] + denoising.content.shape[2:] != (batch, channels):
            msg = "Denoising content must share matching batch and channel dimensions"
            raise ValueError(msg)
        if denoising.ref_logits.shape != (batch, denoising_queries, BOX_COORDINATES):
            msg = "Denoising reference logits must have shape [B, D, 4]"
            raise ValueError(msg)
        total_queries = denoising_queries + matching_queries
        if denoising.attention_mask.shape != (total_queries, total_queries):
            msg = "Denoising attention mask must cover all assembled queries"
            raise ValueError(msg)
        if denoising.attention_mask.dtype != torch.bool:
            msg = "Attention mask must be bool [Q, Q] on the query device"
            raise ValueError(msg)

    def checkpoint_config(self) -> dict[str, object]:
        """Describe the stateless query assembly boundary."""
        return {}
