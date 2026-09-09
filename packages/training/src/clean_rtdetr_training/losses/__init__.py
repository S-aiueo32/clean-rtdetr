# Copyright 2026 Clean RT-DETR contributors
"""Matching losses and assignment geometry."""

from clean_rtdetr_training.losses.criterion import CriterionConfig, SetCriterion
from clean_rtdetr_training.losses.matcher import HungarianMatcher, MatcherConfig
from clean_rtdetr_training.losses.supervision import (
    MatchedPairs,
    MatchedStage,
    gather_matched_pairs,
    iter_matched_stages,
    match_stage,
    matched_box_losses,
    matched_iou_targets,
)

__all__ = [
    "CriterionConfig",
    "HungarianMatcher",
    "MatchedPairs",
    "MatchedStage",
    "MatcherConfig",
    "SetCriterion",
    "gather_matched_pairs",
    "iter_matched_stages",
    "match_stage",
    "matched_box_losses",
    "matched_iou_targets",
]
