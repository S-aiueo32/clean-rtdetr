# Copyright 2026 Clean RT-DETR contributors
"""Small synthetic data boundary for exercising training integrations offline."""

from math import isqrt, lcm
from typing import Literal

import torch
from torch.utils.data import Dataset

from clean_rtdetr import RTDETR
from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.data.contracts import DetectionSample
from clean_rtdetr_training.interfaces import TrainingModel
from clean_rtdetr_training.models.rtdetr import DenoisingTrainingModel


def _resolve_image_size(model: RTDETR, image_size: int | Literal["auto"]) -> int:
    """Resolve square synthetic inputs for the model's stride-aligned feature pyramid."""
    strides = tuple(spec.stride for spec in model.projector.specs)
    alignment = lcm(*strides, *(spec.stride for spec in model.backbone.specs))
    if image_size == "auto":
        positions_per_unit = sum((alignment // stride) ** 2 for stride in strides)
        units = isqrt((model.num_queries - 1) // positions_per_unit) + 1
        # Keep at least 2x2 cells at the coarsest level, including batch-size-one BN.
        return alignment * max(2, units)
    if type(image_size) is not int or image_size <= 0:
        msg = "Synthetic image_size must be 'auto' or a positive integer"
        raise ValueError(msg)
    if image_size % alignment or image_size < 2 * alignment:
        msg = f"Synthetic image_size must be a multiple of {alignment} and at least {2 * alignment}"
        raise ValueError(msg)
    positions = sum((image_size // stride) ** 2 for stride in strides)
    if positions < model.num_queries:
        msg = (
            f"Synthetic image_size provides {positions} memory positions "
            f"for {model.num_queries} queries"
        )
        raise ValueError(msg)
    return image_size


class SyntheticDetectionDataset(Dataset[DetectionSample]):
    """Provide deterministic images with alternating one-object and empty targets.

    This dataset exercises program wiring; it is not a training data pipeline.
    """

    def __init__(self, size: int = 4, image_size: int = 64, num_classes: int = 3) -> None:
        """Configure nonempty synthetic samples without downloading artifacts."""
        if any(type(value) is not int or value <= 0 for value in (size, image_size, num_classes)):
            msg = "Synthetic dataset dimensions and class count must be positive integers"
            raise ValueError(msg)
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    @classmethod
    def from_training_model(
        cls,
        training_model: TrainingModel,
        *,
        size: int = 4,
        image_size: int | Literal["auto"] = "auto",
    ) -> "SyntheticDetectionDataset":
        """Resolve synthetic classes and image size from a constructed DN-wrapped RTDETR.

        Explicit sizes must align with the feature pyramid and supply enough
        memory positions for query selection. Class count comes from the model.
        """
        if not isinstance(training_model, DenoisingTrainingModel) or not isinstance(
            training_model.model, RTDETR
        ):
            msg = "Model-aware synthetic data requires a DenoisingTrainingModel wrapping RTDETR"
            raise TypeError(msg)
        model = training_model.model
        return cls(
            size=size,
            image_size=_resolve_image_size(model, image_size),
            num_classes=model.num_classes,
        )

    def __len__(self) -> int:
        """Return the fixed number of synthetic examples."""
        return self.size

    def __getitem__(self, index: int) -> DetectionSample:
        """Return a normalized image and a typed, variable-length target."""
        if not 0 <= index < self.size:
            raise IndexError(index)
        image = torch.full((3, self.image_size, self.image_size), (index + 1) / (self.size + 1))
        if index % 2:
            target = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
        else:
            target = DetectionTarget(
                torch.tensor([index % self.num_classes]),
                torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            )
        return image, target
