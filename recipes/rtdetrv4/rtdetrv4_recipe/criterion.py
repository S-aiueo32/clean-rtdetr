# Copyright 2026 Clean RT-DETR contributors
"""DSI cosine alignment alongside the installed detection criterion."""

import math
from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional

from clean_rtdetr_training.contracts import DetectionTarget, TrainingOutput
from clean_rtdetr_training.interfaces import TrainingCriterion, TrainingLosses

from .model import DistillationOutput


def alignment_loss(
    student: Float[Tensor, "batch channels height width"],
    teacher: Float[Tensor, "batch channels teacher_height teacher_width"],
) -> Float[Tensor, ""]:
    """Compute upstream RTv4's mean one-minus-cosine over normalized patch tokens.

    Reference: RT-DETRs/RT-DETRv4 at 55fefaa, RTv4Criterion.loss_distillation.
    The teacher is detached here as well as at the training model boundary.
    """
    if student.ndim != 4 or teacher.ndim != 4:  # noqa: PLR2004
        msg = "Alignment features must be NCHW tensors"
        raise ValueError(msg)
    if (
        min(student.shape) <= 0
        or min(teacher.shape) <= 0
        or student.shape[:2] != teacher.shape[:2]
        or student.dtype != teacher.dtype
        or student.device != teacher.device
        or not student.is_floating_point()
    ):
        msg = "Alignment features must share nonempty batch, channels, floating dtype and device"
        raise ValueError(msg)
    teacher = teacher.detach()
    if teacher.shape[-2:] != student.shape[-2:]:
        teacher = functional.interpolate(
            teacher, size=student.shape[-2:], mode="bilinear", align_corners=False
        )
    student_tokens = functional.normalize(student.flatten(2).transpose(1, 2), dim=-1)
    teacher_tokens = functional.normalize(teacher.flatten(2).transpose(1, 2), dim=-1)
    return (1 - functional.cosine_similarity(student_tokens, teacher_tokens, dim=-1)).mean()


class DistillationCriterion(TrainingCriterion[DistillationOutput]):
    """Compose detector losses with a checkpointed, GAM-adjustable DSI weight."""

    distillation_weight: Tensor

    def __init__(
        self,
        detection: TrainingCriterion[TrainingOutput],
        weight: float = 15.0,
    ) -> None:
        """Store mutable weight as a buffer; keep initial configuration immutable."""
        super().__init__()
        if not math.isfinite(weight) or weight <= 0:
            msg = "Distillation weight must be finite and positive"
            raise ValueError(msg)
        self.detection = detection
        self.initial_weight = weight
        self.register_buffer("distillation_weight", torch.tensor(weight, dtype=torch.float32))

    def forward(
        self, output: DistillationOutput, targets: Sequence[DetectionTarget]
    ) -> TrainingLosses:
        """Add DSI once, preserving final, auxiliary and DN detection losses."""
        if output.student is None or output.teacher is None:
            msg = "Distillation loss requires training-time student and teacher features"
            raise ValueError(msg)
        losses = self.detection(output.detection, targets)
        if "loss_distill" in losses:
            msg = "Detection criterion already defines loss_distill"
            raise ValueError(msg)
        losses["loss_distill"] = (
            alignment_loss(output.student, output.teacher) * self.distillation_weight
        )
        return losses

    def checkpoint_config(self) -> dict[str, object]:
        """Record the initial recipe independently from GAM's mutable buffer."""
        return {
            "initial_weight": self.initial_weight,
            "detection_class": (
                f"{type(self.detection).__module__}.{type(self.detection).__qualname__}"
            ),
            "detection": self.detection.checkpoint_config(),
        }
