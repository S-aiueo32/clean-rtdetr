# Copyright 2026 Clean RT-DETR contributors
"""Training composition that owns DN parameters and keeps the core target-free."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr.contracts import DecoderOutput, DecoderPlan
from clean_rtdetr.models.interfaces import Detector
from clean_rtdetr_training.contracts import (
    DenoisingBatch,
    DenoisingOutput,
    DetectionTarget,
    TrainingOutput,
)
from clean_rtdetr_training.interfaces import DenoisingGenerator, TrainingModel


def split_denoising_predictions(
    decoded: DecoderOutput, denoising: DenoisingBatch | None
) -> tuple[DecoderOutput, DenoisingOutput | None]:
    """Separate the leading DN slots from standard decoder logits and boxes.

    Decoder tensors use ``[layers, batch, D+Q, ...]``. All collected layers and
    their autograd connections are retained, and DN metadata is returned unchanged.
    A DN prefix must match the prepared query count and leave matching queries.

    Without DN, the original output is returned by identity, including custom
    ``DecoderOutput`` subclasses. With DN, only an ordinary ``DecoderOutput`` is
    accepted; recipes with extra decoder tensors own their split.
    """
    if denoising is None:
        return decoded, None
    if type(decoded) is not DecoderOutput:
        msg = "Denoising splitting requires standard DecoderOutput; custom outputs own their split"
        raise TypeError(msg)
    denoising.validate()
    count = denoising.metadata.num_queries
    decoded.validate()
    if count >= decoded.logits.shape[2]:
        msg = "Denoising query count must leave matching queries"
        raise ValueError(msg)
    matching = DecoderOutput(decoded.logits[:, :, count:], decoded.boxes[:, :, count:])
    dn_output = DenoisingOutput(
        DecoderOutput(decoded.logits[:, :, :count], decoded.boxes[:, :, :count]),
        denoising.metadata,
    )
    return matching, dn_output


@runtime_checkable
class DenoisingFactory[DenoisingT: DenoisingGenerator](Protocol):
    """Construct a generator using dimensions from the initialized detector."""

    def __call__(self, *, hidden_dim: int, num_classes: int) -> DenoisingT:
        """Return a fully initialized denoising generator."""
        ...


class DenoisingTrainingModel[
    DetectorT: Detector,
    DenoisingT: DenoisingGenerator,
](TrainingModel[TrainingOutput]):
    """Register a detector and DN generator together for optimizers and state dicts."""

    def __init__(self, model: DetectorT, denoising: DenoisingT) -> None:
        """Reject incompatible embeddings before connecting the training stages."""
        super().__init__()
        if denoising.num_classes != model.num_classes or denoising.hidden_dim != model.hidden_dim:
            msg = "Denoising embedding must match model classes and hidden dimension"
            raise ValueError(msg)
        self.model = model
        self.denoising = denoising
        self.train(model.training)

    @classmethod
    def from_factory(
        cls, model: DetectorT, denoising_factory: DenoisingFactory[DenoisingT]
    ) -> "DenoisingTrainingModel[DetectorT, DenoisingT]":
        """Build and register DN parameters after the detector resolves its dimensions."""
        denoising = denoising_factory(hidden_dim=model.hidden_dim, num_classes=model.num_classes)
        return cls(model, denoising)

    def forward(
        self,
        images: Float[Tensor, "batch channels height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
    ) -> TrainingOutput:
        """Build DN during training and expose matching-only final predictions."""
        if not self.training:
            if targets is not None:
                msg = "Targets require training mode"
                raise ValueError(msg)
            return TrainingOutput(self.model(images, plan=plan, collect_aifi=collect_aifi))
        if targets is None or len(targets) != images.shape[0]:
            msg = "Denoising training requires one target per batch image"
            raise ValueError(msg)
        if images.dtype != self.denoising.dtype or images.device != self.denoising.device:
            msg = "Images and denoising embedding must share dtype and device"
            raise ValueError(msg)
        model = self.model
        prepared = self.denoising(targets, num_queries=model.num_queries)
        if prepared is None:
            return TrainingOutput(model(images, plan=plan, collect_aifi=collect_aifi))
        prepared.validate()
        output = model(images, plan=plan, collect_aifi=collect_aifi, denoising=prepared.queries)
        matching, denoising = split_denoising_predictions(output.decoder, prepared)
        return TrainingOutput(replace(output, decoder=matching), denoising)

    def predict(self, images: Float[Tensor, "batch channels height width"]) -> TrainingOutput:
        """Return target-free predictions from the composed detector."""
        return TrainingOutput(self.model(images))

    def checkpoint_config(self) -> dict[str, object]:
        """Describe configuration that is not represented by model parameters."""
        return {
            "denoising": {
                "target": f"{type(self.denoising).__module__}.{type(self.denoising).__qualname__}",
                "config": self.denoising.checkpoint_config(),
            },
            "model": {
                "target": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "config": self.model.checkpoint_config(),
            },
        }
