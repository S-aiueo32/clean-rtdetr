# Copyright 2026 Clean RT-DETR contributors
"""Regression coverage for distributed set-loss normalization."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from tempfile import mkstemp
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed

from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr_training import DetectionTarget, SetCriterion
from clean_rtdetr_training.contracts import MatchIndices
from clean_rtdetr_training.interfaces import Matcher

if TYPE_CHECKING:
    from collections.abc import Sequence


class _PrefixMatcher(Matcher):
    """Match each target to the correspondingly positioned query."""

    def forward(
        self, predictions: DetectionOutput, targets: Sequence[DetectionTarget]
    ) -> tuple[MatchIndices, ...]:
        return tuple(
            MatchIndices(
                torch.arange(target.labels.numel(), device=predictions.logits.device),
                torch.arange(target.labels.numel(), device=predictions.logits.device),
            )
            for target in targets
        )

    def checkpoint_config(self) -> dict[str, object]:
        return {"type": "prefix"}


def _targets(count: int, *, offset: int = 0) -> list[DetectionTarget]:
    labels = torch.arange(offset, offset + count, dtype=torch.int64)
    boxes = torch.full((count, 4), 0.25, dtype=torch.float64)
    return [DetectionTarget(labels, boxes)]


def _worker(rank: int, world_size: int, rendezvous: str, queue: mp.Queue) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=f"file://{rendezvous}"
    )
    logits = torch.tensor(
        [
            [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    boxes = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.3, 0.3], [0.7, 0.7, 0.1, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = _targets(2) if rank == 0 else _targets(0)
    losses = SetCriterion(matcher=_PrefixMatcher()).stage_losses(
        DetectionOutput(logits, boxes), targets
    )
    loss = torch.stack(tuple(losses.values())).sum()
    loss.backward()
    logits_gradient = logits.grad
    boxes_gradient = boxes.grad
    assert logits_gradient is not None
    assert boxes_gradient is not None
    for gradient in (logits_gradient, boxes_gradient):
        torch.distributed.all_reduce(gradient)
        gradient.div_(world_size)
    reduced_loss = loss.detach().clone()
    torch.distributed.all_reduce(reduced_loss)
    reduced_loss /= world_size
    if rank == 0:
        queue.put((reduced_loss.item(), logits_gradient.tolist(), boxes_gradient.tolist()))
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed unavailable")
def test_two_rank_loss_and_gradients_match_concatenated_batch() -> None:
    """A rank with no targets still matches the global concatenated objective."""
    baseline_logits = torch.tensor(
        [
            [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]],
            [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    baseline_boxes = torch.tensor(
        [
            [[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.3, 0.3], [0.7, 0.7, 0.1, 0.1]],
            [[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.3, 0.3], [0.7, 0.7, 0.1, 0.1]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    baseline_targets = _targets(2) + _targets(0)
    baseline_losses = SetCriterion(matcher=_PrefixMatcher()).stage_losses(
        DetectionOutput(baseline_logits, baseline_boxes), baseline_targets
    )
    baseline_loss = torch.stack(tuple(baseline_losses.values())).sum()
    baseline_loss.backward()

    descriptor, rendezvous = mkstemp(prefix="clean-rtdetr-ddp-")
    os.close(descriptor)
    Path(rendezvous).unlink()
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, 2, rendezvous, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        result = queue.get(timeout=60)
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
    for process in processes:
        assert process.exitcode == 0
    reduced_loss, reduced_logits_grad, reduced_boxes_grad = result
    torch.testing.assert_close(
        torch.tensor(reduced_loss, dtype=baseline_loss.dtype), baseline_loss.detach()
    )
    assert baseline_logits.grad is not None
    assert baseline_boxes.grad is not None
    expected_logits_grad = baseline_logits.grad.sum(dim=0)
    expected_boxes_grad = baseline_boxes.grad.sum(dim=0)
    torch.testing.assert_close(
        torch.tensor(reduced_logits_grad, dtype=expected_logits_grad.dtype)[0], expected_logits_grad
    )
    torch.testing.assert_close(
        torch.tensor(reduced_boxes_grad, dtype=expected_boxes_grad.dtype)[0], expected_boxes_grad
    )
