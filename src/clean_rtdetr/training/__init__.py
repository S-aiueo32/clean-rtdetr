# Copyright 2026 Clean RT-DETR contributors
"""Single-process matching and baseline losses; SciPy is loaded only for assignment."""

from clean_rtdetr.training.contracts import DetectionTarget, MatchIndices
from clean_rtdetr.training.criterion import CriterionConfig, SetCriterion
from clean_rtdetr.training.matcher import HungarianMatcher, MatcherConfig

__all__ = [
    "CriterionConfig",
    "DetectionTarget",
    "HungarianMatcher",
    "MatchIndices",
    "MatcherConfig",
    "SetCriterion",
]
