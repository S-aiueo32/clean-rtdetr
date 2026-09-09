# Copyright 2026 Clean RT-DETR contributors
"""Detection batch collation."""

from collections.abc import Sequence

import torch

from clean_rtdetr_training.data.contracts import DetectionBatch, DetectionSample


def collate_detection_batch(samples: Sequence[DetectionSample]) -> DetectionBatch:
    """Stack equally sized images and preserve variable-length targets."""
    return torch.stack([image for image, _ in samples]), [target for _, target in samples]
