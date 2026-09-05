# Copyright 2026 Clean RT-DETR contributors
"""Stable module boundaries for replacing RT-DETR stages in experiments."""

from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import (
    DecoderInput,
    DecoderOutput,
    DecoderPlan,
    DenoisingBatch,
    DenoisingQueries,
    EncodedFeatures,
    FeatureSpec,
    HybridEncoderOutput,
    ModelOutput,
    MultiScaleFeatures,
    QuerySet,
    SelectionOutput,
)


class Stage(nn.Module, ABC):
    """Base for replaceable stages with explicit checkpoint provenance."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Return serializable construction provenance for this stage."""
        raise NotImplementedError


class Backbone(Stage):
    """Extract ordered multi-scale features from images."""

    specs: tuple[FeatureSpec, ...]

    @abstractmethod
    def forward(self, x: Float[Tensor, "batch 3 height width"]) -> MultiScaleFeatures:
        """Extract ordered multi-scale features from a batch."""
        raise NotImplementedError


class Encoder(Stage):
    """Encode and fuse multi-scale backbone features."""

    specs: tuple[FeatureSpec, ...]
    output_specs: tuple[FeatureSpec, ...]
    hidden_dim: int

    @abstractmethod
    def forward(
        self, features: MultiScaleFeatures, *, collect_aifi: bool = False
    ) -> HybridEncoderOutput:
        """Encode features and optionally retain AIFI output."""
        raise NotImplementedError


class Projector(Stage):
    """Project feature levels into decoder memory."""

    specs: tuple[FeatureSpec, ...]
    hidden_dim: int

    @abstractmethod
    def forward(self, features: MultiScaleFeatures) -> EncodedFeatures:
        """Project feature levels into flattened encoded memory."""
        raise NotImplementedError


class Selector(Stage):
    """Select and initialize decoder queries from encoded memory."""

    hidden_dim: int
    num_classes: int
    num_queries: int

    @abstractmethod
    def forward(self, features: EncodedFeatures) -> SelectionOutput:
        """Select top scoring query positions."""
        raise NotImplementedError


class Assembler(Stage):
    """Assemble matching and optional denoising decoder queries."""

    @abstractmethod
    def forward(
        self, matching: QuerySet, denoising: DenoisingQueries | None = None
    ) -> DecoderInput:
        """Assemble decoder input query tensors."""
        raise NotImplementedError


class Decoder(Stage):
    """Iteratively decode query features against encoded memory."""

    hidden_dim: int
    num_classes: int
    num_levels: int

    @abstractmethod
    def forward(
        self,
        queries: DecoderInput,
        features: EncodedFeatures,
        *,
        plan: DecoderPlan | None = None,
    ) -> DecoderOutput:
        """Return predictions for the requested decoder prefix."""
        raise NotImplementedError


class Detector(nn.Module, ABC):
    """Image-to-prediction boundary used by training integrations."""

    num_classes: int
    hidden_dim: int
    num_queries: int

    @abstractmethod
    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
        denoising: DenoisingBatch | None = None,
    ) -> ModelOutput:
        """Return matching predictions and optional denoising predictions."""
        raise NotImplementedError

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Return serializable detector construction provenance."""
        raise NotImplementedError
