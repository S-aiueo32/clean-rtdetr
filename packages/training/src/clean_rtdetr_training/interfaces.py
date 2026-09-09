# Copyright 2026 Clean RT-DETR contributors
"""Module interfaces shared by training integrations and detection implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import DenoisingBatch, DetectionOutput
from clean_rtdetr_training.contracts import DetectionTarget, MatchIndices

type TrainingLosses = dict[str, Float[Tensor, ""]]


class DenoisingGenerator(nn.Module, ABC):
    """Prepare denoising queries without exposing their parameterization."""

    num_classes: int
    hidden_dim: int

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Floating-point dtype required by the generated query content."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device required by the generated query content."""

    @abstractmethod
    def forward(
        self, targets: Sequence[DetectionTarget], *, num_queries: int
    ) -> DenoisingBatch | None:
        """Return prepared queries and assignments, or disable DN for this batch."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe non-tensor settings affecting query preparation."""


class Matcher(nn.Module, ABC):
    """Assign prediction indices to target indices independently for each image."""

    @abstractmethod
    def forward(
        self, predictions: DetectionOutput, targets: Sequence[DetectionTarget]
    ) -> tuple[MatchIndices, ...]:
        """Validate inputs and return one assignment per image on the prediction device."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe non-tensor assignment settings needed for compatible resumes."""


class TrainingModel[PredictionT](nn.Module, ABC):
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
    ) -> PredictionT:
        """Return the prediction payload consumed by the paired criterion."""

    @abstractmethod
    def predict(self, images: Float[Tensor, "batch 3 height width"]) -> PredictionT:
        """Return target-free predictions while preserving the caller's train/eval mode."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe non-tensor settings with primitives, sequences, and mappings.

        Include settings that affect continuation but are absent from state_dict.
        Model parameters and mutable tensor state belong in state_dict instead.
        """


class TrainingCriterion[PredictionT](nn.Module, ABC):
    """Consume a model's prediction payload and return weighted scalar losses."""

    @abstractmethod
    def forward(self, output: PredictionT, targets: Sequence[DetectionTarget]) -> TrainingLosses:
        """Return differentiable scalar terms whose sum is the optimization objective."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Describe loss settings needed to reject incompatible checkpoint resumes."""
