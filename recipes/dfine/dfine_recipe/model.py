# Copyright 2026 Clean RT-DETR contributors
"""Training wrapper and factory for the recipe-local FDR hybrid."""

from collections.abc import Sequence

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr import create_model
from clean_rtdetr.contracts import DecoderPlan, ModelOutput
from clean_rtdetr.models import RTDETR
from clean_rtdetr.models.interfaces import Detector
from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.interfaces import TrainingModel

from .decoder import FDRDecoder


class DFINETrainingModel(TrainingModel[ModelOutput]):
    """Use the public RT-DETR stages with a recipe-local FDR decoder.

    This bounded recipe does not generate denoising queries. Targets are
    accepted for compatibility with the Lightning training boundary and are
    consumed by the paired criterion instead.
    """

    def __init__(self, model: Detector) -> None:
        """Register an RT-DETR model whose decoder is :class:`FDRDecoder`."""
        super().__init__()
        if not isinstance(model, RTDETR) or not isinstance(model.decoder, FDRDecoder):
            msg = "DFINETrainingModel requires an RTDETR with an FDRDecoder"
            raise TypeError(msg)
        self.model = model

    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
    ) -> ModelOutput:
        """Return predictions; target validation belongs to the paired criterion."""
        del targets
        return self.model(images, plan=plan, collect_aifi=collect_aifi)

    def predict(self, images: Float[Tensor, "batch 3 height width"]) -> ModelOutput:
        """Return target-free predictions while preserving the caller's train mode."""
        was_training = self.training
        try:
            self.eval()
            return self.model(images)
        finally:
            self.train(was_training)

    def checkpoint_config(self) -> dict[str, object]:
        """Record the wrapped model's architecture and recipe wrapper."""
        model_type = type(self.model)
        return {
            "model": {
                "target": f"{model_type.__module__}.{model_type.__qualname__}",
                "config": self.model.checkpoint_config(),
            },
            "denoising": None,
        }


def create_training_model(
    name: str = "rtdetr_r18vd",
    *,
    num_classes: int = 3,
    reg_max: int = 32,
    reg_scale: float = 4.0,
    up: float = 0.5,
) -> DFINETrainingModel:
    """Build an uninitialized RT-DETR + D-FINE FDR hybrid.

    This is a recipe composition, not a full D-FINE architecture or a
    checkpoint-compatible implementation of the upstream D-FINE repository.
    """
    base = create_model(name, num_classes=num_classes)
    decoder = FDRDecoder(
        base.decoder.config,
        reg_max=reg_max,
        reg_scale=reg_scale,
        up=up,
        default_eval_plan=base.decoder.default_eval_plan,
    )
    model = RTDETR(
        base.backbone,
        base.encoder,
        base.projector,
        base.selector,
        base.assembler,
        decoder,
    )
    return DFINETrainingModel(model)
