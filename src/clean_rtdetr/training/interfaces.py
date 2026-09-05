# Copyright 2026 Clean RT-DETR contributors
"""Module interfaces shared by training integrations and detection implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.training.contracts import DetectionTarget

PredictionT_co = TypeVar("PredictionT_co", covariant=True)
PredictionT_contra = TypeVar("PredictionT_contra", contravariant=True)
TrainingLosses = dict[str, Float[Tensor, ""]]


class TrainingModel(nn.Module, ABC, Generic[PredictionT_co]):
    """Define training, inference, and configuration without prescribing a model layout.

    Subclasses remain ordinary PyTorch modules: forward calls retain hooks,
    parameters are registered, and state_dict includes all learnable state.
    """

    @abstractmethod
    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
    ) -> PredictionT_co:
        """Return the prediction payload consumed by the paired criterion."""

    @abstractmethod
    def predict(self, images: Float[Tensor, "batch 3 height width"]) -> PredictionT_co:
        """Return target-free predictions while preserving the caller's train/eval mode."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe non-tensor settings with primitives, sequences, and mappings.

        Include settings that affect continuation but are absent from state_dict.
        Model parameters and mutable tensor state belong in state_dict instead.
        """


class TrainingCriterion(nn.Module, ABC, Generic[PredictionT_contra]):
    """Consume a model's prediction payload and return weighted scalar losses."""

    @abstractmethod
    def forward(
        self, output: PredictionT_contra, targets: Sequence[DetectionTarget]
    ) -> TrainingLosses:
        """Return differentiable scalar terms whose sum is the optimization objective."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe loss settings needed to reject incompatible checkpoint resumes."""
