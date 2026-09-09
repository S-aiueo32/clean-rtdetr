# Copyright 2026 Clean RT-DETR contributors
"""Composable image-to-prediction RT-DETR model."""

from jaxtyping import Float
from torch import Tensor

from clean_rtdetr.core.contracts import (
    QUERY_RANK,
    DecoderOutput,
    DecoderPlan,
    DenoisingBatch,
    DenoisingOutput,
    DetectionOutput,
    ModelOutput,
)
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder
from clean_rtdetr.models.encoder import HybridEncoder
from clean_rtdetr.models.interfaces import (
    Assembler,
    Backbone,
    Detector,
    Encoder,
    Projector,
    Selector,
)
from clean_rtdetr.models.interfaces import (
    Decoder as DecoderInterface,
)
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.selector import QuerySelector


class RTDETR[
    BackboneT: Backbone,
    EncoderT: Encoder,
    ProjectorT: Projector,
    SelectorT: Selector,
    AssemblerT: Assembler,
    DecoderT: DecoderInterface,
](Detector):
    """Compose the explicit backbone, encoder, query, and decoder boundaries."""

    def __init__(  # noqa: PLR0913, PLR0917 - explicit stage injection is the public API
        self,
        backbone: BackboneT,
        encoder: EncoderT,
        projector: ProjectorT,
        selector: SelectorT,
        assembler: AssemblerT,
        decoder: DecoderT,
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
        if self.selector.hidden_dim != hidden_dim or self.decoder.hidden_dim != hidden_dim:
            msg = "Selector and decoder hidden dimensions must match encoder"
            raise ValueError(msg)
        if self.selector.num_classes != self.decoder.num_classes:
            msg = "Selector and decoder class counts must match"
            raise ValueError(msg)
        if self.decoder.num_levels != len(encoder_outputs):
            msg = "Decoder level count must match encoder outputs"
            raise ValueError(msg)

    def forward(
        self,
        images: Float[Tensor, "batch channels height width"],
        *,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
        denoising: DenoisingBatch | None = None,
    ) -> ModelOutput:
        """Run images through all stages and return decoder and auxiliary predictions.

        The stage boundaries and retained outputs follow this data flow::

            images → backbone → encoder ─┬─ features → projector → memory
                                         └─ after_aifi (optional)

            memory → selector ─┬─ matching queries ─┐
                               └─ encoder_aux       ├→ assembler → queries
            prepared DN queries (optional) ─────────┘

            memory ──┐
                     ├→ decoder → split ─┬─ matching outputs
            queries ─┘                   └─ DN outputs (optional)

        ``after_aifi`` is optional; encoder auxiliary predictions keep the
        selector's gradient path. Prepared DN queries enter at the assembler and
        their leading decoder slots are separated after all requested layers run.
        ``plan`` controls decoder depth and retained intermediate predictions.
        This boundary returns raw logits and normalized cxcywh boxes.
        """
        self._validate_denoising(denoising)
        features = self.backbone(images)
        encoded_features = self.encoder(features, collect_aifi=collect_aifi)
        encoded = self.projector(encoded_features.features)
        selected = self.selector(encoded)
        queries = self.assembler(selected.queries, None if denoising is None else denoising.queries)
        decoded = self.decoder(queries, encoded, plan=plan)
        matching, dn_output = self._split_predictions(decoded, denoising)
        return ModelOutput(
            decoder=matching,
            encoder_aux=DetectionOutput(selected.encoder_logits, selected.encoder_boxes),
            after_aifi=encoded_features.after_aifi,
            denoising=dn_output,
        )

    def _validate_denoising(self, denoising: DenoisingBatch | None) -> None:
        """Check DN query counts before executing any model stage."""
        if denoising is not None:
            count = denoising.metadata.num_queries
            if (
                type(count) is not int
                or count <= 0
                or denoising.queries.content.ndim != QUERY_RANK
                or count != denoising.queries.content.shape[1]
            ):
                msg = "Denoising metadata query count must be a positive integer matching queries"
                raise ValueError(msg)

    def _split_predictions(
        self, decoded: DecoderOutput, denoising: DenoisingBatch | None
    ) -> tuple[DecoderOutput, DenoisingOutput | None]:
        """Separate the leading DN predictions from the matching query predictions.

        Decoder tensors have layout ``[layers, B, D+Q, ...]``. Slice ``:D`` for
        denoising and ``D:`` for matching, retaining every collected layer. Slices
        preserve autograd connections; DN metadata travels with its predictions.
        A DN prefix must leave at least one matching query. Without DN, the
        original ``DecoderOutput`` is returned directly.
        """
        if denoising is None:
            matching = decoded
            dn_output = None
        else:
            count = denoising.metadata.num_queries
            if count <= 0 or count >= decoded.logits.shape[2]:
                msg = "Denoising query count must leave matching queries"
                raise ValueError(msg)
            matching = DecoderOutput(decoded.logits[:, :, count:], decoded.boxes[:, :, count:])
            dn_output = DenoisingOutput(
                DecoderOutput(decoded.logits[:, :, :count], decoded.boxes[:, :, :count]),
                denoising.metadata,
            )
        return matching, dn_output

    @property
    def num_classes(self) -> int:
        """Return the detector's foreground class count."""
        return self.decoder.num_classes

    @property
    def hidden_dim(self) -> int:
        """Return the detector's shared hidden channel width."""
        return self.decoder.hidden_dim

    @property
    def num_queries(self) -> int:
        """Return the detector's selected query count."""
        return self.selector.num_queries

    def checkpoint_config(self) -> dict[str, object]:
        """Describe injected stages and their non-tensor construction settings."""
        return {
            name: {
                "target": f"{type(stage).__module__}.{type(stage).__qualname__}",
                "config": stage.checkpoint_config(),
            }
            for name, stage in (
                ("backbone", self.backbone),
                ("encoder", self.encoder),
                ("projector", self.projector),
                ("selector", self.selector),
                ("assembler", self.assembler),
                ("decoder", self.decoder),
            )
        }


type BaselineRTDETR = RTDETR[
    PResNet, HybridEncoder, MemoryProjector, QuerySelector, QueryAssembler, Decoder
]
