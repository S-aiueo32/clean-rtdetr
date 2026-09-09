# Copyright 2026 Clean RT-DETR contributors
# ruff: noqa: EM101, TRY003
"""A small CNN auxiliary head and a lightweight TAL-style criterion.

This is a recipe-local auxiliary branch.  It intentionally implements the
compact task-aligned assignment used by this recipe, rather than claiming
PP-YOLOE/ATSS training parity.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr import DetectionOutput, FeatureSpec, MultiScaleFeatures
from clean_rtdetr_training import DetectionTarget
from clean_rtdetr_training.contracts import validate_detection
from clean_rtdetr_training.losses.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou


@dataclass(frozen=True)
class CNNOutput:
    """Dense CNN predictions and the geometry used to decode them."""

    detection: DetectionOutput
    distribution_logits: Float[Tensor, "batch locations 4 bins"]
    anchor_points: Float[Tensor, "locations 2"]
    stride_scales: Float[Tensor, "locations 2"]


class CNNAuxiliaryHead(nn.Module):
    """Predict dense classes and positive ltrb distributions at each level."""

    def __init__(self, specs: tuple[FeatureSpec, ...], num_classes: int, reg_max: int = 16) -> None:
        """Build one trunk and two prediction projections per feature level."""
        super().__init__()
        if not specs:
            raise ValueError("CNN auxiliary head requires at least one feature spec")
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if type(reg_max) is not int or reg_max <= 0:
            raise ValueError("reg_max must be a positive integer")
        self.specs = tuple(specs)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.trunks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(spec.channels, spec.channels, 3, padding=1),
                nn.SiLU(),
            )
            for spec in self.specs
        )
        self.score_heads = nn.ModuleList(
            nn.Conv2d(spec.channels, num_classes, 1) for spec in self.specs
        )
        self.distribution_heads = nn.ModuleList(
            nn.Conv2d(spec.channels, 4 * (reg_max + 1), 1) for spec in self.specs
        )

    def checkpoint_config(self) -> dict[str, object]:
        """Return every construction setting needed to identify this branch."""
        return {
            "specs": [asdict(spec) for spec in self.specs],
            "num_classes": self.num_classes,
            "reg_max": self.reg_max,
            "architecture": "per_level_3x3_conv_silu_shared_trunk_1x1_heads",
        }

    def forward(self, features: MultiScaleFeatures) -> CNNOutput:
        """Decode one dense detection stream from the supplied feature levels."""
        features.validate()
        if features.specs != self.specs:
            raise ValueError("Input FeatureSpec differs from the CNN head construction spec")
        class_logits: list[Tensor] = []
        distributions: list[Tensor] = []
        anchors: list[Tensor] = []
        scales: list[Tensor] = []
        for level, trunk, score, distribution in zip(
            features.levels,
            self.trunks,
            self.score_heads,
            self.distribution_heads,
            strict=True,
        ):
            _, _, height, width = level.shape
            hidden = trunk(level)
            class_logits.append(
                score(hidden).permute(0, 2, 3, 1).reshape(level.shape[0], -1, self.num_classes)
            )
            distributions.append(
                distribution(hidden)
                .reshape(level.shape[0], 4, self.reg_max + 1, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(level.shape[0], -1, 4, self.reg_max + 1)
            )
            y, x = torch.meshgrid(
                torch.arange(height, device=level.device, dtype=level.dtype),
                torch.arange(width, device=level.device, dtype=level.dtype),
                indexing="ij",
            )
            stride = level.new_tensor([1.0 / width, 1.0 / height])
            anchors.append((torch.stack((x, y), -1) + 0.5).reshape(-1, 2) * stride)
            scales.append(stride.expand(height * width, 2))
        logits = torch.cat(class_logits, dim=1)
        distribution_logits = torch.cat(distributions, dim=1)
        anchor_points = torch.cat(anchors)
        stride_scales = torch.cat(scales)
        bins = torch.arange(
            self.reg_max + 1, device=distribution_logits.device, dtype=distribution_logits.dtype
        )
        scale4 = torch.cat((stride_scales, stride_scales), dim=-1)
        distances = distribution_logits.softmax(-1).matmul(bins) * scale4[None, :, :]
        left_top = anchor_points[None, :, :] - distances[..., :2]
        right_bottom = anchor_points[None, :, :] + distances[..., 2:]
        # Clip corners before converting to cxcywh so the output remains a
        # valid normalized box without changing its center/size independently.
        boxes_xyxy = torch.cat((left_top, right_bottom), -1).clamp(0, 1)
        boxes = torch.cat(
            (
                (boxes_xyxy[..., :2] + boxes_xyxy[..., 2:]) / 2,
                boxes_xyxy[..., 2:] - boxes_xyxy[..., :2],
            ),
            -1,
        )
        return CNNOutput(
            DetectionOutput(logits, boxes), distribution_logits, anchor_points, stride_scales
        )


@dataclass(frozen=True)
class _Assignment:
    indices: Tensor
    targets: Tensor
    quality: Tensor


class CNNCriterion(nn.Module):
    """Task-aligned dense supervision for :class:`CNNAuxiliaryHead`."""

    def __init__(  # noqa: PLR0913 - each setting is checkpointed explicitly
        self,
        *,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        weight_vfl: float = 1.0,
        weight_iou: float = 2.5,
        weight_dfl: float = 0.5,
    ) -> None:
        """Configure TAL assignment and weighted VFL, GIoU, and DFL terms."""
        super().__init__()
        if type(topk) is not int or topk <= 0:
            raise ValueError("topk must be a positive integer")
        values = (alpha, beta, weight_vfl, weight_iou, weight_dfl)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("CNN criterion settings must be finite and nonnegative")
        if not any((weight_vfl, weight_iou, weight_dfl)):
            raise ValueError("CNN criterion loss weights cannot all be zero")
        self.topk, self.alpha, self.beta = topk, alpha, beta
        self.weight_vfl, self.weight_iou, self.weight_dfl = weight_vfl, weight_iou, weight_dfl

    def checkpoint_config(self) -> dict[str, object]:
        """Return all assignment and loss settings for checkpoint metadata."""
        return {
            "topk": self.topk,
            "alpha": self.alpha,
            "beta": self.beta,
            "weight_vfl": self.weight_vfl,
            "weight_iou": self.weight_iou,
            "weight_dfl": self.weight_dfl,
            "assignment": "center_inside_topk_alignment_iou_conflict_resolution",
            "iou_loss": "generalized_iou",
        }

    def forward(
        self, output: CNNOutput, targets: Sequence[DetectionTarget]
    ) -> dict[str, Float[Tensor, ""]]:
        """Compute weighted VFL, GIoU, and distribution focal losses."""
        validate_detection(output.detection, targets)
        batch, locations, classes = output.detection.logits.shape
        distributions = output.distribution_logits
        if distributions.ndim != 4 or distributions.shape[:3] != (batch, locations, 4):  # noqa: PLR2004
            raise ValueError("Distribution logits must have shape [B, S, 4, R+1]")
        if output.anchor_points.shape != (locations, 2) or output.stride_scales.shape != (
            locations,
            2,
        ):
            raise ValueError("Anchor points and stride scales must have shape [S, 2]")
        if (
            distributions.shape[-1] < 2  # noqa: PLR2004 - minimum bin count
            or not torch.isfinite(distributions).all()
        ):
            raise ValueError("Distribution logits must be finite with at least two bins")
        if (
            not distributions.is_floating_point()
            or distributions.dtype != output.detection.logits.dtype
            or distributions.device != output.detection.logits.device
            or output.anchor_points.ndim != 2  # noqa: PLR2004
            or output.stride_scales.ndim != 2  # noqa: PLR2004
            or not output.anchor_points.is_floating_point()
            or not output.stride_scales.is_floating_point()
            or output.anchor_points.dtype != output.detection.logits.dtype
            or output.stride_scales.dtype != output.detection.logits.dtype
            or output.anchor_points.device != output.detection.logits.device
            or output.stride_scales.device != output.detection.logits.device
            or not torch.isfinite(output.anchor_points).all()
            or not torch.isfinite(output.stride_scales).all()
            or bool((output.stride_scales <= 0).any())
        ):
            raise ValueError("Auxiliary geometry must be finite, positive, and compatible")
        assignments = [self._assign(output, target, image) for image, target in enumerate(targets)]
        normalizer = sum(
            (a.quality.sum() for a in assignments), output.detection.logits.new_zeros(())
        )
        normalizer = normalizer.clamp(min=1)
        quality = output.detection.logits.new_zeros(batch, locations, classes)
        positive = torch.zeros_like(quality, dtype=torch.bool)
        for image, assignment in enumerate(assignments):
            if assignment.indices.numel():
                labels = targets[image].labels[assignment.targets]
                quality[image, assignment.indices, labels] = assignment.quality
                positive[image, assignment.indices, labels] = True
        probability = output.detection.logits.sigmoid().detach()
        weight = self._vfl_weight(quality, positive, probability)
        vfl = (
            functional.binary_cross_entropy_with_logits(
                output.detection.logits, quality, weight=weight, reduction="sum"
            )
            / normalizer
        )
        iou = output.detection.logits.sum() * 0
        dfl = output.distribution_logits.sum() * 0
        for image, assignment in enumerate(assignments):
            if assignment.indices.numel():
                predicted = output.detection.boxes[image, assignment.indices]
                target_boxes = targets[image].boxes[assignment.targets]
                giou = torch.diag(
                    generalized_box_iou(cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target_boxes))
                )
                iou = iou + ((1 - giou) * assignment.quality).sum()
                dfl = dfl + self._distribution_loss(output, image, assignment, target_boxes)
        return {
            "loss_vfl": vfl * self.weight_vfl,
            "loss_iou": iou / normalizer * self.weight_iou,
            "loss_dfl": dfl / normalizer * self.weight_dfl,
        }

    @staticmethod
    def _vfl_weight(quality: Tensor, positive: Tensor, probability: Tensor) -> Tensor:
        """Use IoU quality for positives and focal negatives for all other classes."""
        return torch.where(positive, quality, probability.pow(2) * 0.75)

    @torch.no_grad()
    def _assign(self, output: CNNOutput, target: DetectionTarget, image: int = 0) -> _Assignment:
        """Build detached center-filtered top-k TAL assignments for one image."""
        locations = output.anchor_points.shape[0]
        if not target.labels.numel():
            empty = torch.empty(0, dtype=torch.int64, device=output.anchor_points.device)
            return _Assignment(empty, empty, output.detection.logits.new_empty(0))
        anchors = output.anchor_points
        target_xyxy = cxcywh_to_xyxy(target.boxes)
        inside = (
            (anchors[:, None, 0] >= target_xyxy[None, :, 0])
            & (anchors[:, None, 1] >= target_xyxy[None, :, 1])
            & (anchors[:, None, 0] <= target_xyxy[None, :, 2])
            & (anchors[:, None, 1] <= target_xyxy[None, :, 3])
        )
        predicted = output.detection.boxes[image].detach()
        ious = box_iou(cxcywh_to_xyxy(predicted), target_xyxy)[0].clamp(min=0).detach()
        probabilities = output.detection.logits[image].detach().sigmoid()[:, target.labels]
        alignment = probabilities.pow(self.alpha) * ious.pow(self.beta)
        alignment = alignment.masked_fill(~inside, -1)
        selected = torch.zeros(locations, dtype=torch.bool, device=anchors.device)
        selected_targets = torch.full((locations,), -1, dtype=torch.int64, device=anchors.device)
        selected_iou = torch.zeros(
            locations, dtype=output.detection.logits.dtype, device=anchors.device
        )
        for gt in range(target.labels.numel()):
            candidates = torch.where(inside[:, gt])[0]
            if candidates.numel() == 0:
                continue
            count = min(self.topk, candidates.numel())
            order = torch.argsort(alignment[candidates, gt], descending=True, stable=True)[:count]
            chosen = candidates[order]
            for location in chosen.tolist():
                current = ious[location, gt]
                if (not selected[location]) or current > selected_iou[location]:
                    selected[location] = True
                    selected_targets[location] = gt
                    selected_iou[location] = current
        indices = torch.where(selected)[0]
        assigned_targets = selected_targets[indices]
        assigned_alignment = alignment[indices, assigned_targets].clamp(min=0)
        max_alignment = torch.zeros_like(assigned_alignment)
        max_iou = torch.zeros_like(assigned_alignment)
        for gt in range(target.labels.numel()):
            mask = assigned_targets == gt
            if mask.any():
                max_alignment[mask] = assigned_alignment[mask].max()
                max_iou[mask] = selected_iou[indices][mask].max()
        quality = assigned_alignment / max_alignment.clamp(min=1e-12) * max_iou
        return _Assignment(indices, assigned_targets, quality.clamp(0, 1))

    @staticmethod
    def _distribution_loss(
        output: CNNOutput, image: int, assignment: _Assignment, target_boxes: Tensor
    ) -> Tensor:
        """Apply two-bin interpolation to target ltrb distances in stride units."""
        points = output.anchor_points[assignment.indices]
        scales = output.stride_scales[assignment.indices]
        corners = cxcywh_to_xyxy(target_boxes)
        distances = torch.cat((points - corners[:, :2], corners[:, 2:] - points), -1) / torch.cat(
            (scales, scales), dim=-1
        )
        reg_max = output.distribution_logits.shape[-1] - 1
        distances = distances.clamp(0, reg_max - 1e-6)
        lower = distances.floor().long()
        upper = (lower + 1).clamp(max=reg_max)
        upper_weight = distances - lower
        logits = output.distribution_logits[image, assignment.indices]
        log_probability = logits.log_softmax(-1)
        loss = -(
            log_probability.gather(-1, lower.unsqueeze(-1)).squeeze(-1) * (1 - upper_weight)
            + log_probability.gather(-1, upper.unsqueeze(-1)).squeeze(-1) * upper_weight
        )
        return (loss.mean(-1) * assignment.quality).sum()
