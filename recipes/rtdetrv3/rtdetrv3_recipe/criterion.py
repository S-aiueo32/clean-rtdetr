# Copyright 2026 Clean RT-DETR contributors
"""Average O2O groups and add repeated-target O2M and CNN supervision."""

import math
from collections.abc import Sequence

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr_training import HungarianMatcher, SetCriterion
from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.interfaces import TrainingCriterion

from .cnn import CNNCriterion
from .model import RTDETRv3Output


class RTDETRv3Criterion(TrainingCriterion[RTDETRv3Output]):
    """Combine O2O, repeated-target O2M, and CNN training losses."""

    def __init__(
        self,
        *,
        repeats: int = 4,
        weight_o2o: float = 1.0,
        weight_o2m: float = 1.0,
        weight_cnn: float = 1.0,
    ) -> None:
        """Configure repeated targets and the three supervision weights."""
        super().__init__()
        if type(repeats) is not int or repeats <= 0:
            msg = "repeats must be a positive integer"
            raise ValueError(msg)
        weights = (weight_o2o, weight_o2m, weight_cnn)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            msg = "Branch weights must be finite, nonnegative, and not all zero"
            raise ValueError(msg)
        self.repeats = repeats
        self.weight_o2o = weight_o2o
        self.weight_o2m = weight_o2m
        self.weight_cnn = weight_cnn
        self.detection = SetCriterion(matcher=HungarianMatcher())
        self.cnn = CNNCriterion()

    def forward(
        self, output: RTDETRv3Output, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Average weighted O2O losses and add O2M and CNN terms.

        O2O includes the main and SAP groups, preserving their DN supervision.
        O2M repeats targets before matching and normalization. Returned terms
        already include branch weights; evaluation-only outputs are rejected.
        """
        if output.o2m is None or output.cnn is None:
            msg = "RTDETRv3Criterion requires training branch outputs"
            raise ValueError(msg)
        groups = (output.main, *output.sap)
        losses = {}
        for index, group in enumerate(groups):
            losses.update(
                (f"{name}_o2o_{index}", value * self.weight_o2o / len(groups))
                for name, value in self.detection(group, targets).items()
            )
        repeated = [
            DetectionTarget(
                target.labels.repeat(self.repeats), target.boxes.repeat(self.repeats, 1)
            )
            for target in targets
        ]
        losses.update(
            (f"{name}_o2m", value * self.weight_o2m)
            for name, value in self.detection(output.o2m, repeated).items()
        )
        losses.update(
            (f"{name}_cnn", value * self.weight_cnn)
            for name, value in self.cnn(output.cnn, targets).items()
        )
        return losses

    def checkpoint_config(self) -> dict[str, object]:
        """Record all loss and assignment choices affecting continuation."""
        return {
            "repeats": self.repeats,
            "weight_o2o": self.weight_o2o,
            "weight_o2m": self.weight_o2m,
            "weight_cnn": self.weight_cnn,
            "detection": self.detection.checkpoint_config(),
            "cnn": self.cnn.checkpoint_config(),
        }
