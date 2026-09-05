# Copyright 2026 Clean RT-DETR contributors
"""Explicit composition of the R50vd image-to-prediction path."""

from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import DetectionOutput, ModelOutput
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig, DecoderPlan
from clean_rtdetr.models.encoder import HybridEncoder
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig


class RTDETR(nn.Module):
    """Own the modules connecting RGB images to unprocessed decoder predictions.

    Inputs follow the official RGB float [0, 1] convention, without mean/std
    normalization. This module does not resize, pad, normalize, or postprocess.
    """

    def __init__(
        self,
        backbone: PResNet,
        encoder: HybridEncoder,
        projector: MemoryProjector,
        selector: QuerySelector,
        decoder: Decoder,
    ) -> None:
        """Register components once and reject incompatible feature boundaries."""
        super().__init__()
        if backbone.specs != encoder.specs or encoder.output_specs != projector.specs:
            msg = "Backbone, encoder, and projector feature specifications must agree"
            raise ValueError(msg)
        if (
            projector.hidden_dim != selector.config.hidden_dim
            or projector.hidden_dim != decoder.config.hidden_dim
            or selector.config.num_classes != decoder.config.num_classes
            or len(projector.specs) != decoder.config.num_levels
        ):
            msg = "Projector, selector, and decoder configurations are incompatible"
            raise ValueError(msg)
        self.backbone = backbone
        self.encoder = encoder
        self.projector = projector
        self.selector = selector
        self.decoder = decoder
        self.assembler = QueryAssembler()

    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
    ) -> ModelOutput:
        """Run each stage, preserving the selector/assembler gradient boundaries."""
        features = self.backbone(images)
        encoded = self.encoder(features, collect_aifi=collect_aifi)
        memory = self.projector(encoded.features)
        selected = self.selector(memory)
        queries = self.assembler(selected.queries)
        predictions = self.decoder(queries, memory, plan=plan)
        return ModelOutput(
            predictions,
            DetectionOutput(selected.encoder_logits, selected.encoder_boxes),
            encoded.after_aifi,
        )


def build_rtdetr_r50vd() -> RTDETR:
    """Build the official 80-class, 300-query, six-layer architecture without weights.

    Modules start in training mode like ordinary PyTorch modules. Call ``eval()``
    for inference and use the explicit compatibility loader for official weights.
    """
    backbone = PResNet()
    encoder = HybridEncoder(backbone.specs)
    projector = MemoryProjector(encoder.output_specs)
    selector = QuerySelector(QuerySelectorConfig())
    decoder = Decoder(DecoderConfig())
    return RTDETR(backbone, encoder, projector, selector, decoder)
