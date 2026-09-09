# Copyright 2026 Clean RT-DETR contributors
"""Training composition that owns DN parameters and keeps the core target-free."""

from collections.abc import Sequence

from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import DecoderOutput, DenoisingOutput, DetectionOutput, ModelOutput
from clean_rtdetr.models.decoder import DecoderPlan
from clean_rtdetr.models.rtdetr import RTDETR
from clean_rtdetr.training.contracts import DetectionTarget
from clean_rtdetr.training.denoising import DenoisingQueryGenerator


class DenoisingTrainingModel(nn.Module):
    """Register a detector and DN generator together for optimizers and state dicts."""

    def __init__(self, model: RTDETR, denoising: DenoisingQueryGenerator) -> None:
        """Reject incompatible embeddings before connecting the training stages."""
        super().__init__()
        config = model.decoder.config
        if (
            denoising.class_embed.num_embeddings != config.num_classes + 1
            or denoising.class_embed.embedding_dim != config.hidden_dim
        ):
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
        if (
            images.dtype != self.denoising.class_embed.weight.dtype
            or images.device != self.denoising.class_embed.weight.device
        ):
            msg = "Images and denoising embedding must share dtype and device"
            raise ValueError(msg)
        model = self.model
        prepared = self.denoising(targets, num_queries=model.selector.config.num_queries)
        if prepared is None:
            return model(images, plan=plan, collect_aifi=collect_aifi)
        features = model.backbone(images)
        encoded_features = model.encoder(features, collect_aifi=collect_aifi)
        encoded = model.projector(encoded_features.features)
        selected = model.selector(encoded)
        queries = model.assembler(selected.queries, prepared.queries)
        decoded = model.decoder(queries, encoded, plan=plan)
        count = prepared.metadata.num_queries
        return ModelOutput(
            decoder=DecoderOutput(decoded.logits[:, :, count:], decoded.boxes[:, :, count:]),
            encoder_aux=DetectionOutput(selected.encoder_logits, selected.encoder_boxes),
            after_aifi=encoded_features.after_aifi,
            denoising=DenoisingOutput(
                DecoderOutput(decoded.logits[:, :, :count], decoded.boxes[:, :, :count]),
                prepared.metadata,
            ),
        )
