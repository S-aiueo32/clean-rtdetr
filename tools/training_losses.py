# Copyright 2026 Clean RT-DETR contributors
"""Compute baseline matching losses and prediction gradients on synthetic CPU data."""

import torch

from clean_rtdetr.core.contracts import DetectionOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion


def main() -> None:
    """Exercise a mixed batch with one empty image and no downloaded artifacts."""
    torch.manual_seed(41)
    logits = torch.randn(2, 5, 3, requires_grad=True)
    box_logits = torch.randn(2, 5, 4, requires_grad=True)
    predictions = DetectionOutput(logits, box_logits.sigmoid())
    targets = [
        DetectionTarget(
            labels=torch.tensor([0, 2], dtype=torch.int64),
            boxes=torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.1, 0.2]]),
        ),
        DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4)),
    ]
    losses = SetCriterion(matcher=HungarianMatcher())(predictions, targets)
    total = torch.stack(list(losses.values())).sum()
    total.backward()
    print({name: float(loss.detach()) for name, loss in losses.items()})
    print(f"Total weighted loss: {float(total.detach()):.6f}")
    for name, tensor in (("logits", logits), ("box_logits", box_logits)):
        gradient = tensor.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            msg = f"Expected finite {name} gradients"
            raise RuntimeError(msg)
        print(f"{name} gradient norm: {float(gradient.norm()):.6f}")


if __name__ == "__main__":
    main()
