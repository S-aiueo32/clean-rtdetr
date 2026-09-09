# Copyright 2026 Clean RT-DETR contributors
"""RT-DETRv3-inspired training branches composed through public model stages."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr import create_model
from clean_rtdetr.contracts import (
    DecoderInput,
    DetectionOutput,
    EncodedFeatures,
    ModelOutput,
    QuerySet,
)
from clean_rtdetr.geometry import generate_anchors
from clean_rtdetr.models.rtdetr import BaselineRTDETR
from clean_rtdetr.models.selector import QuerySelector
from clean_rtdetr_training import DenoisingConfig, DenoisingQueryGenerator
from clean_rtdetr_training.contracts import DetectionTarget, TrainingOutput
from clean_rtdetr_training.interfaces import TrainingModel
from clean_rtdetr_training.models.rtdetr import split_denoising_predictions

from .cnn import CNNAuxiliaryHead, CNNOutput


@dataclass(frozen=True)
class RTDETRv3Output:
    """Keep inference predictions separate from training-only supervision."""

    main: TrainingOutput
    sap: tuple[TrainingOutput, ...] = ()
    o2m: TrainingOutput | None = None
    cnn: CNNOutput | None = None

    @property
    def final(self) -> DetectionOutput:
        """Return only the ordinary one-to-one branch for detection."""
        return self.main.final


def perturb_queries(queries: DecoderInput, probability: float) -> DecoderInput:
    """Randomly block matching-query edges while preserving the DN mask.

    The same mask is reused across decoder layers, batches, and heads. Entirely
    blocked rows retain their diagonal edge to keep tiny-query experiments finite.
    """
    if not math.isfinite(probability) or not 0 <= probability < 1:
        msg = "SAP probability must be finite and in [0, 1)"
        raise ValueError(msg)
    queries.validate()
    total = queries.content.shape[1]
    count = queries.matching_queries
    mask = (
        torch.zeros(total, total, dtype=torch.bool, device=queries.content.device)
        if queries.attention_mask is None
        else queries.attention_mask.clone()
    )
    blocked = torch.rand(count, count, device=mask.device) < probability
    empty_rows = blocked.all(dim=-1).nonzero().flatten()
    blocked[empty_rows, empty_rows] = False
    mask[-count:, -count:] = blocked
    return replace(queries, attention_mask=mask)


class RTDETRv3TrainingModel(TrainingModel[RTDETRv3Output]):
    """Share the backbone, encoder, and decoder across independently selected groups."""

    def __init__(
        self,
        model: BaselineRTDETR,
        *,
        sap_groups: int = 3,
        mask_probability: float = 0.1,
        o2m_queries: int = 450,
        num_denoising: int = 4,
    ) -> None:
        """Register independent selectors, shared query mapping, DN, and CNN heads."""
        super().__init__()
        if type(sap_groups) is not int or sap_groups < 0:
            msg = "sap_groups must be a nonnegative integer"
            raise ValueError(msg)
        if type(o2m_queries) is not int or o2m_queries <= 0:
            msg = "o2m_queries must be a positive integer"
            raise ValueError(msg)
        if not math.isfinite(mask_probability) or not 0 <= mask_probability < 1:
            msg = "SAP probability must be finite and in [0, 1)"
            raise ValueError(msg)
        if model.selector.config.learned_content:
            msg = "This recipe requires feature-initialized queries"
            raise ValueError(msg)
        self.model = model
        self.mask_probability = mask_probability
        self.sap_selectors = nn.ModuleList(
            QuerySelector(model.selector.config) for _ in range(sap_groups)
        )
        self.o2m_selector = QuerySelector(replace(model.selector.config, num_queries=o2m_queries))
        projection = nn.Linear(model.hidden_dim, model.hidden_dim)
        nn.init.xavier_uniform_(projection.weight)
        self.map_memory = nn.Sequential(projection, nn.LayerNorm(model.hidden_dim))
        self.denoising = DenoisingQueryGenerator(
            hidden_dim=model.hidden_dim,
            num_classes=model.num_classes,
            config=DenoisingConfig(num_denoising=num_denoising),
        )
        self.cnn = CNNAuxiliaryHead(model.encoder.output_specs, model.num_classes)
        self.train(model.training)

    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
    ) -> RTDETRv3Output:
        """Run shared image stages once and supervise isolated query groups."""
        if not self.training:
            if targets is not None:
                msg = "Targets require training mode"
                raise ValueError(msg)
            return RTDETRv3Output(TrainingOutput(self.model(images)))
        if targets is None or len(targets) != images.shape[0]:
            msg = "Training requires one target per image"
            raise ValueError(msg)
        features = self.model.encoder(self.model.backbone(images)).features
        encoded = self.model.projector(features)
        shapes = tuple(tuple(shape) for shape in encoded.spatial_shapes.tolist())
        _, valid = generate_anchors(
            shapes, device=images.device, eps=self.model.selector.config.anchor_eps
        )
        mapped = self.map_memory((encoded.memory * valid).detach())
        main = self._group(self.model.selector, encoded, targets=targets)
        sap = tuple(
            self._group(
                cast("QuerySelector", selector),
                encoded,
                targets=targets,
                mapped=mapped,
                perturb=True,
            )
            for selector in self.sap_selectors
        )
        o2m = self._group(self.o2m_selector, encoded, mapped=mapped)
        return RTDETRv3Output(main, sap, o2m, self.cnn(features))

    def _group(
        self,
        selector: QuerySelector,
        encoded: EncodedFeatures,
        *,
        targets: Sequence[DetectionTarget] | None = None,
        mapped: Tensor | None = None,
        perturb: bool = False,
    ) -> TrainingOutput:
        """Decode one isolated group with the same decoder and prediction heads."""
        selected = selector(encoded)
        matching = selected.queries
        if mapped is not None:
            content = mapped.gather(
                1, matching.source_indices.unsqueeze(-1).expand(-1, -1, mapped.shape[-1])
            )
            matching = QuerySet(content, matching.ref_logits, matching.source_indices)
        prepared = (
            None if targets is None else self.denoising(targets, num_queries=selector.num_queries)
        )
        queries = self.model.assembler(matching, None if prepared is None else prepared.queries)
        if perturb:
            queries = perturb_queries(queries, self.mask_probability)
        decoded = self.model.decoder(queries, encoded)
        decoded, denoising = split_denoising_predictions(decoded, prepared)
        return TrainingOutput(
            ModelOutput(
                decoded,
                DetectionOutput(selected.encoder_logits, selected.encoder_boxes),
                None,
            ),
            denoising,
        )

    def predict(self, images: Float[Tensor, "batch 3 height width"]) -> RTDETRv3Output:
        """Execute only the baseline detector and restore all caller module modes."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return RTDETRv3Output(TrainingOutput(self.model(images)))
        finally:
            for module, training in modes:
                module.training = training

    def checkpoint_config(self) -> dict[str, object]:
        """Record branch, perturbation, DN, and architecture settings for resume."""
        return {
            "model": self.model.checkpoint_config(),
            "sap_selectors": [
                cast("QuerySelector", selector).checkpoint_config()
                for selector in self.sap_selectors
            ],
            "o2m_selector": self.o2m_selector.checkpoint_config(),
            "mask_probability": self.mask_probability,
            "denoising": self.denoising.checkpoint_config(),
            "cnn": self.cnn.checkpoint_config(),
        }


def create_training_model(  # noqa: PLR0913 - expose branch controls to Hydra
    name: str = "rtdetr_r18vd",
    *,
    num_classes: int = 1,
    sap_groups: int = 3,
    mask_probability: float = 0.1,
    o2m_queries: int = 450,
    num_denoising: int = 4,
) -> RTDETRv3TrainingModel:
    """Build a randomly initialized CPU recipe using the installed baseline model."""
    return RTDETRv3TrainingModel(
        create_model(name, num_classes=num_classes),
        sap_groups=sap_groups,
        mask_probability=mask_probability,
        o2m_queries=o2m_queries,
        num_denoising=num_denoising,
    )
