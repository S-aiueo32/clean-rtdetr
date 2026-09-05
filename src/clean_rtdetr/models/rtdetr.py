# Copyright 2026 Clean RT-DETR contributors
"""Composable image-to-prediction RT-DETR model."""

from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.core.contracts import DetectionOutput, ModelOutput
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig, DecoderPlan
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig


class RTDETR(nn.Module):
    """Compose the explicit backbone, encoder, query, and decoder boundaries."""

    def __init__(  # noqa: PLR0913, PLR0917 - explicit stage injection is the public API
        self,
        backbone: PResNet,
        encoder: HybridEncoder,
        projector: MemoryProjector,
        selector: QuerySelector,
        assembler: QueryAssembler,
        decoder: Decoder,
    ) -> None:
        """Register components and reject incompatible stage contracts early."""
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.projector = projector
        self.selector = selector
        self.assembler = assembler
        self.decoder = decoder
        self._validate_components()

    def _validate_components(self) -> None:
        """Check metadata and dimensions shared across injected components."""
        if self.encoder.specs != self.backbone.specs:
            msg = "Backbone and encoder feature specs must match"
            raise ValueError(msg)
        encoder_outputs = self.encoder.output_specs
        if encoder_outputs != self.projector.specs:
            msg = "Encoder and projector feature specs must match"
            raise ValueError(msg)
        hidden_dim = self.encoder.hidden_dim
        if self.projector.hidden_dim != hidden_dim:
            msg = "Projector hidden dimension must match encoder"
            raise ValueError(msg)
        selector_config = self.selector.config
        decoder_config = self.decoder.config
        if selector_config.hidden_dim != hidden_dim or decoder_config.hidden_dim != hidden_dim:
            msg = "Selector and decoder hidden dimensions must match encoder"
            raise ValueError(msg)
        if selector_config.num_classes != decoder_config.num_classes:
            msg = "Selector and decoder class counts must match"
            raise ValueError(msg)
        if decoder_config.num_levels != len(encoder_outputs):
            msg = "Decoder level count must match encoder outputs"
            raise ValueError(msg)

    def forward(
        self,
        images: Float[Tensor, "batch channels height width"],
        *,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
    ) -> ModelOutput:
        """Run images through all stages and return decoder and auxiliary predictions."""
        features = self.backbone(images)
        encoded_features = self.encoder(features, collect_aifi=collect_aifi)
        encoded = self.projector(encoded_features.features)
        selected = self.selector(encoded)
        queries = self.assembler(selected.queries)
        decoded = self.decoder(queries, encoded, plan=plan)
        return ModelOutput(
            decoder=decoded,
            encoder_aux=DetectionOutput(selected.encoder_logits, selected.encoder_boxes),
            after_aifi=encoded_features.after_aifi,
        )


def build_rtdetr_r50vd() -> RTDETR:
    """Build the baseline R50vd RT-DETR composition without loading weights."""
    backbone = PResNet()
    encoder = HybridEncoder(backbone.specs, HybridEncoderConfig(hidden_dim=256))
    projector = MemoryProjector(encoder.output_specs, hidden_dim=256)
    selector = QuerySelector(
        QuerySelectorConfig(hidden_dim=256, num_classes=80, num_queries=300, learned_content=False)
    )
    decoder = Decoder(
        DecoderConfig(
            hidden_dim=256,
            num_classes=80,
            num_layers=6,
            num_heads=8,
            num_levels=3,
        )
    )
    return RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)
