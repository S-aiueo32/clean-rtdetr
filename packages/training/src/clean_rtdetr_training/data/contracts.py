# Copyright 2026 Clean RT-DETR contributors
"""Shared detection dataset and batch contracts."""

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr_training.contracts import DetectionTarget

DetectionSample = tuple[Float[Tensor, "3 height width"], DetectionTarget]
DetectionBatch = tuple[Float[Tensor, "batch 3 height width"], list[DetectionTarget]]
