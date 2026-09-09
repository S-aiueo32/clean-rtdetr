# Copyright 2026 Clean RT-DETR contributors
"""Small synthetic data boundary for exercising training integrations offline."""

import torch
from torch.utils.data import Dataset

from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.data.contracts import DetectionSample


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
