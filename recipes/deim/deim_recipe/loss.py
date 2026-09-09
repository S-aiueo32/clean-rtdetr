# Copyright 2026 Clean RT-DETR contributors
"""Matchability-aware classification from the pinned DEIM MAL reference.

The equation follows ``DEIMCriterion.loss_labels_mal`` at commit
``09d35d53d39ee3145a1e61e3a989b28b9468d1dd``; see ``../NOTICE.md``. This
implementation adapts the equation to Clean RT-DETR tensors and leaves loss
weighting to the recipe criterion.
"""

import math

from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional


class MatchabilityAwareLoss(nn.Module):
    """Use IoU**gamma soft targets and unit weight for matched classes.

    Unmatched classes receive detached sigmoid(logits)**gamma weight. Unlike
    VFL, even a matched class with zero IoU has unit weight. This implements
    DEIM MAL with mal_alpha=None; it does not implement Dense O2O augmentation.
    """

    def __init__(self, gamma: float = 1.5) -> None:
        """Set the positive finite exponent used by DEIM's RT-DETR recipe."""
        super().__init__()
        if not math.isfinite(gamma) or gamma <= 0:
            msg = "MAL gamma must be finite and positive"
            raise ValueError(msg)
        self.gamma = gamma

    def forward(
        self,
        logits: Float[Tensor, "batch queries classes"],
        target_scores: Float[Tensor, "batch queries classes"],
        positive_mask: Bool[Tensor, "batch queries classes"],
        normalizer: int,
    ) -> Float[Tensor, ""]:
        """Return an unweighted scalar; criterion owns its configured multiplier."""
        scores = target_scores.detach().pow(self.gamma)
        weight = logits.detach().sigmoid().pow(self.gamma)
        weight = weight.masked_fill(positive_mask, 1.0)
        loss = functional.binary_cross_entropy_with_logits(
            logits, scores, weight=weight, reduction="none"
        )
        return loss.mean(1).sum() * logits.shape[1] / normalizer
