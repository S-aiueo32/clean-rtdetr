# Copyright 2026 Clean RT-DETR contributors
"""Single-process matching and baseline losses; SciPy is loaded only for assignment."""

from clean_rtdetr_training.configuration import compose_training_config
from clean_rtdetr_training.contracts import (
    DenoisingMetadata,
    DenoisingOutput,
    DetectionTarget,
    MatchIndices,
    TrainingOutput,
)
from clean_rtdetr_training.interfaces import (
    DenoisingGenerator,
    Matcher,
    TrainingCriterion,
    TrainingLosses,
    TrainingModel,
)
from clean_rtdetr_training.losses.criterion import CriterionConfig, SetCriterion
from clean_rtdetr_training.losses.matcher import HungarianMatcher, MatcherConfig
from clean_rtdetr_training.models import DenoisingBatch, DenoisingConfig, DenoisingQueryGenerator
from clean_rtdetr_training.models.rtdetr import DenoisingTrainingModel, split_denoising_predictions

__all__ = [
    "CriterionConfig",
    "DenoisingBatch",
    "DenoisingConfig",
    "DenoisingGenerator",
    "DenoisingMetadata",
    "DenoisingOutput",
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
    "TrainingOutput",
    "compose_training_config",
    "split_denoising_predictions",
]
