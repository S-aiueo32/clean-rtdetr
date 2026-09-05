# Copyright 2026 Clean RT-DETR contributors
"""Optional Lightning integration; import this namespace only with the lightning extra."""

from clean_rtdetr.training.lightning.ema import RTDETREMA
from clean_rtdetr.training.lightning.module import RTDETRLightningModule
from clean_rtdetr.training.lightning.optim import RTDETRParameterGroups
from clean_rtdetr.training.lightning.state import EpochBoundaryState

__all__ = ["RTDETREMA", "EpochBoundaryState", "RTDETRLightningModule", "RTDETRParameterGroups"]
