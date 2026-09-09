# Copyright 2026 Clean RT-DETR contributors
"""Single-process matching and baseline losses; SciPy is loaded only for assignment."""

from clean_rtdetr_training.configuration import compose_training_config
from clean_rtdetr_training.contracts import DetectionTarget, MatchIndices
from clean_rtdetr_training.denoising import DenoisingBatch, DenoisingConfig, DenoisingQueryGenerator
from clean_rtdetr_training.denoising.model import DenoisingTrainingModel
from clean_rtdetr_training.interfaces import (
    DenoisingGenerator,
    Matcher,
    TrainingCriterion,
    TrainingLosses,
    TrainingModel,
)
from clean_rtdetr_training.losses.criterion import CriterionConfig, SetCriterion
from clean_rtdetr_training.losses.matcher import HungarianMatcher, MatcherConfig

__all__ = [
    "CriterionConfig",
    "DenoisingBatch",
    "DenoisingConfig",
    "DenoisingGenerator",
    "DenoisingQueryGenerator",
    "DenoisingTrainingModel",
    "DetectionTarget",
    "HungarianMatcher",
    "MatchIndices",
    "Matcher",
    "MatcherConfig",
    "SetCriterion",
    "TrainingCriterion",
    "TrainingLosses",
    "TrainingModel",
    "compose_training_config",
]
