# Copyright 2026 Clean RT-DETR contributors
"""Hungarian matching for D-FINE distribution-refined predictions."""

from clean_rtdetr_training.losses.matcher import HungarianMatcher, MatcherConfig


class DFINEMatcher(HungarianMatcher):
    """Match raw D-FINE boxes without applying normalized-box validation."""

    def __init__(self, config: MatcherConfig | None = None) -> None:
        """Create a matcher with optional Hungarian cost settings."""
        super().__init__(config, prediction_box_range="unbounded")


__all__ = ["DFINEMatcher", "MatcherConfig"]
