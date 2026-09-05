# Copyright 2026 Clean RT-DETR contributors
"""Single-process matching and baseline losses; SciPy is loaded only for assignment."""

from clean_rtdetr.training.contracts import DetectionTarget, MatchIndices
from clean_rtdetr.training.criterion import CriterionConfig, SetCriterion
from clean_rtdetr.training.denoising import DenoisingBatch, DenoisingConfig, DenoisingQueryGenerator
from clean_rtdetr.training.interfaces import TrainingCriterion, TrainingLosses, TrainingModel
from clean_rtdetr.training.matcher import HungarianMatcher, MatcherConfig
from clean_rtdetr.training.model import DenoisingTrainingModel

__all__ = [
    "CriterionConfig",
    "DenoisingBatch",
    "DenoisingConfig",
    "DenoisingQueryGenerator",
    "DenoisingTrainingModel",
    "DetectionTarget",
    "HungarianMatcher",
    "MatchIndices",
    "MatcherConfig",
    "SetCriterion",
    "TrainingCriterion",
    "TrainingLosses",
    "TrainingModel",
]
