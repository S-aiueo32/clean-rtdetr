# Copyright 2026 Clean RT-DETR contributors
"""Training composition that owns DN parameters and keeps the core target-free."""

from collections.abc import Sequence

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr.core.contracts import DecoderPlan, ModelOutput
from clean_rtdetr.models.interfaces import Detector
from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.interfaces import DenoisingGenerator, TrainingModel


class DenoisingTrainingModel[DetectorT: Detector, DenoisingT: DenoisingGenerator](
    TrainingModel[ModelOutput]
):
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

    def forward(
        self,
        images: Float[Tensor, "batch channels height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
    ) -> ModelOutput:
        """Build DN during training and expose matching-only final predictions."""
        if not self.training:
            if targets is not None:
                msg = "Targets require training mode"
                raise ValueError(msg)
            return self.model(images, plan=plan, collect_aifi=collect_aifi)
        if targets is None or len(targets) != images.shape[0]:
            msg = "Denoising training requires one target per batch image"
            raise ValueError(msg)
        if images.dtype != self.denoising.dtype or images.device != self.denoising.device:
            msg = "Images and denoising embedding must share dtype and device"
            raise ValueError(msg)
        model = self.model
        prepared = self.denoising(targets, num_queries=model.num_queries)
        if prepared is None:
            return model(images, plan=plan, collect_aifi=collect_aifi)
        if prepared.metadata.num_queries != prepared.queries.content.shape[1]:
            msg = "Denoising metadata query count must match prepared queries"
            raise ValueError(msg)
        return model(images, plan=plan, collect_aifi=collect_aifi, denoising=prepared)

    def predict(self, images: Float[Tensor, "batch channels height width"]) -> ModelOutput:
        """Return target-free predictions from the composed detector."""
        return self.model(images)

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
