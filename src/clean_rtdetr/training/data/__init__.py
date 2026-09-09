# Copyright 2026 Clean RT-DETR contributors
"""Detection data contracts, synthetic dataset, and collation."""

from clean_rtdetr.training.data.collate import collate_detection_batch
from clean_rtdetr.training.data.contracts import DetectionBatch, DetectionSample
from clean_rtdetr.training.data.synthetic import SyntheticDetectionDataset

__all__ = [
    "DetectionBatch",
    "DetectionSample",
    "SyntheticDetectionDataset",
    "collate_detection_batch",
]
