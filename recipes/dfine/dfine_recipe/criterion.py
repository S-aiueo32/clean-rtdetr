# Copyright 2026 Clean RT-DETR contributors
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""VFL, union localization and final-layer distillation; see ../NOTICE.md."""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from torch.nn import functional

from clean_rtdetr.contracts import DetectionOutput, ModelOutput
from clean_rtdetr_training import DetectionTarget, Matcher, MatchIndices, TrainingCriterion
from clean_rtdetr_training.losses.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou
from dfine_recipe.decoder import DistributionOutput
from dfine_recipe.distribution import box_to_distances, interpolate_bins, weighting_function


@dataclass(frozen=True)
class DFINECriterionConfig:
    """Configure distribution coordinates, weighted objectives and temperature."""

    reg_max: int = 32
    reg_scale: float = 4.0
    up: float = 0.5
    temperature: float = 5.0
    weight_vfl: float = 1.0
    weight_bbox: float = 5.0
    weight_giou: float = 2.0
    weight_fgl: float = 0.15
    weight_ddf: float = 1.5
    alpha: float = 0.75
    gamma: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid distribution geometry and unusable loss settings."""
        weighting_function(self.reg_max, self.up, self.reg_scale, torch.zeros(1))
        if not all(math.isfinite(value) for value in asdict(self).values()):
            msg = "D-FINE loss parameters must be finite"
            raise ValueError(msg)
        weights = (
            self.weight_vfl,
            self.weight_bbox,
            self.weight_giou,
            self.weight_fgl,
            self.weight_ddf,
        )
        if min(weights) < 0 or not any(weights):
            msg = "Loss weights must be nonnegative and not all zero"
            raise ValueError(msg)
        if self.temperature <= 0 or not 0 <= self.alpha <= 1 or self.gamma < 0:
            msg = "Invalid temperature or focal parameters"
            raise ValueError(msg)


def union_matches(stages: Sequence[Sequence[MatchIndices]]) -> tuple[MatchIndices, ...]:
    """Vote over assignments; each query keeps its most frequent target.

    Multiple queries may supervise the same target. Equal counts follow sorted
    pair order, made stable here rather than relying on upstream's unstable sort.
    """
    result = []
    for image in zip(*stages, strict=True):
        pairs = torch.cat([torch.stack((m.prediction_indices, m.target_indices), 1) for m in image])
        unique, counts = torch.unique(pairs, return_counts=True, dim=0)
        chosen: dict[int, int] = {}
        for pair in unique[torch.argsort(counts, descending=True, stable=True)].tolist():
            chosen.setdefault(pair[0], pair[1])
        result.append(
            MatchIndices(
                torch.tensor(list(chosen), dtype=torch.int64, device=pairs.device),
                torch.tensor(list(chosen.values()), dtype=torch.int64, device=pairs.device),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class Pairs:
    """Aligned predictions and targets for one assignment."""

    batch: Int64[Tensor, " matches"]
    query: Int64[Tensor, " matches"]
    labels: Int64[Tensor, " matches"]
    predicted: Float[Tensor, "matches 4"]
    target: Float[Tensor, "matches 4"]

    @property
    def iou(self) -> Float[Tensor, " matches"]:
        """Return detached localization quality of aligned boxes."""
        return torch.diag(
            box_iou(cxcywh_to_xyxy(self.predicted), cxcywh_to_xyxy(self.target))[0]
        ).detach()


def gather_pairs(
    stage: DetectionOutput, targets: Sequence[DetectionTarget], matches: Sequence[MatchIndices]
) -> Pairs:
    """Gather assignments while retaining prediction box gradients."""
    batch = torch.cat([torch.full_like(m.prediction_indices, i) for i, m in enumerate(matches)])
    query = torch.cat([m.prediction_indices for m in matches])
    return Pairs(
        batch,
        query,
        torch.cat([t.labels[m.target_indices] for t, m in zip(targets, matches, strict=True)]),
        stage.boxes[batch, query],
        torch.cat([t.boxes[m.target_indices] for t, m in zip(targets, matches, strict=True)]),
    )


class DFINECriterion(TrainingCriterion[ModelOutput]):
    """Supervise refined distributions through the public training interface."""

    def __init__(self, matcher: Matcher, config: DFINECriterionConfig | None = None) -> None:
        """Inject assignment and immutable loss settings."""
        super().__init__()
        self.matcher = matcher
        self.config = DFINECriterionConfig() if config is None else config

    def checkpoint_config(self) -> dict[str, object]:
        """Record loss settings and matcher identity for checkpoint resumes."""
        return {
            "config": asdict(self.config),
            "matcher": self.matcher.checkpoint_config(),
            "matcher_class": f"{type(self.matcher).__module__}.{type(self.matcher).__qualname__}",
        }

    def forward(
        self, output: ModelOutput, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Match stages once, then apply independent VFL and union localization."""
        decoded = self._validate(output)
        count = decoded.logits.shape[0]
        stages = [
            DetectionOutput(logits, boxes)
            for logits, boxes in zip(decoded.logits, decoded.boxes, strict=True)
        ]
        stages.extend((decoded.pre, output.encoder_aux))
        matches = [self.matcher(stage, targets) for stage in stages]
        union = union_matches(matches)
        normalizer = max(sum(t.labels.numel() for t in targets), 1)
        union_normalizer = max(sum(m.prediction_indices.numel() for m in union), 1)
        suffixes = [f"_aux_{i}" for i in range(count - 1)] + ["", "_pre", "_enc"]
        losses = {}
        for index, (stage, assignment, suffix) in enumerate(
            zip(stages, matches, suffixes, strict=True)
        ):
            pairs = gather_pairs(stage, targets, union)
            current = self._boxes(pairs, union_normalizer)
            current["loss_vfl"] = self._classification(
                stage, gather_pairs(stage, targets, assignment), normalizer
            )
            if index < count:
                current.update(self._local(decoded, index, pairs, union_normalizer))
            losses.update((name + suffix, value) for name, value in current.items())
        return losses

    def _validate(self, output: ModelOutput) -> DistributionOutput:
        """Check distribution geometry and the supported execution mode."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            msg = "D-FINE recipe supports single-process normalization only"
            raise ValueError(msg)
        decoded = output.decoder
        if not isinstance(decoded, DistributionOutput):
            msg = "DFINECriterion requires DistributionOutput from FDRDecoder"
            raise TypeError(msg)
        decoded.validate()
        if (
            decoded.corners.shape != (*decoded.logits.shape[:3], 4, self.config.reg_max + 1)
            or decoded.reg_scale != self.config.reg_scale
            or decoded.up != self.config.up
        ):
            msg = "Decoder and criterion distribution settings must match"
            raise ValueError(msg)
        return decoded

    def _classification(
        self, stage: DetectionOutput, pairs: Pairs, normalizer: int
    ) -> Float[Tensor, ""]:
        """Use independent one-to-one VFL assignments with detached IoU targets."""
        scores = torch.zeros_like(stage.logits)
        positive = torch.zeros_like(stage.logits, dtype=torch.bool)
        scores[pairs.batch, pairs.query, pairs.labels] = pairs.iou
        positive[pairs.batch, pairs.query, pairs.labels] = True
        weight = (
            self.config.alpha * stage.logits.detach().sigmoid().pow(self.config.gamma) * (~positive)
            + scores
        )
        loss = functional.binary_cross_entropy_with_logits(
            stage.logits, scores, weight=weight, reduction="sum"
        )
        return loss / normalizer * self.config.weight_vfl

    def _boxes(self, pairs: Pairs, normalizer: int) -> dict[str, Float[Tensor, ""]]:
        """Apply union assignments to L1 and GIoU."""
        bbox = functional.l1_loss(pairs.predicted, pairs.target, reduction="sum")
        giou = (
            1
            - torch.diag(
                generalized_box_iou(cxcywh_to_xyxy(pairs.predicted), cxcywh_to_xyxy(pairs.target))
            )
        ).sum()
        return {
            "loss_bbox": bbox / normalizer * self.config.weight_bbox,
            "loss_giou": giou / normalizer * self.config.weight_giou,
        }

    def _local(
        self, decoded: DistributionOutput, layer: int, pairs: Pairs, normalizer: int
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute IoU-weighted bin CE and final-layer distribution distillation."""
        corners = decoded.corners[layer]
        bins = weighting_function(
            self.config.reg_max, self.config.up, self.config.reg_scale, corners
        )
        references = decoded.references[pairs.batch, pairs.query].detach()
        distances = box_to_distances(
            references, cxcywh_to_xyxy(pairs.target), self.config.reg_scale
        )
        left, right = interpolate_bins(distances, bins)
        flat = corners[pairs.batch, pairs.query].reshape(-1, self.config.reg_max + 1)
        ce = functional.cross_entropy(flat, left.flatten(), reduction="none") * (
            1 - right.flatten()
        )
        ce += functional.cross_entropy(flat, left.flatten() + 1, reduction="none") * right.flatten()
        fgl = (ce.reshape(-1, 4) * pairs.iou[:, None]).sum() / normalizer
        return {
            "loss_fgl": fgl * self.config.weight_fgl,
            "loss_ddf": self._distillation(decoded, layer, pairs) * self.config.weight_ddf,
        }

    def _distillation(
        self, decoded: DistributionOutput, layer: int, pairs: Pairs
    ) -> Float[Tensor, ""]:
        """Balance positive and negative KL using square-root counts."""
        student = decoded.corners[layer]
        if layer == decoded.corners.shape[0] - 1 or torch.equal(student, decoded.corners[-1]):
            return student.sum() * 0
        temperature = self.config.temperature
        teacher = decoded.corners[-1].detach()
        kl = (
            functional.kl_div(
                functional.log_softmax(student / temperature, dim=-1),
                functional.softmax(teacher / temperature, dim=-1),
                reduction="none",
            ).sum(-1)
            * temperature**2
        )
        quality = decoded.logits[-1].detach().sigmoid().amax(-1).clone()
        quality[pairs.batch, pairs.query] = pairs.iou
        positive = torch.zeros_like(quality, dtype=torch.bool)
        positive[pairs.batch, pairs.query] = True
        weighted = kl * quality[..., None]
        # The shared four-corner and 8/B factors cancel in upstream's ratio.
        num_pos = int(positive.sum())
        num_neg = positive.numel() - num_pos
        pos = weighted[positive].mean() if num_pos else student.sum() * 0
        neg = weighted[~positive].mean() if num_neg else student.sum() * 0
        return (pos * math.sqrt(num_pos) + neg * math.sqrt(num_neg)) / (
            math.sqrt(num_pos) + math.sqrt(num_neg)
        )
