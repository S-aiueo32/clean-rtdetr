# Copyright 2026 Clean RT-DETR contributors
"""Training-only semantic supervision through the public AIFI output."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from clean_rtdetr.contracts import DetectionOutput
from clean_rtdetr.models import RTDETR
from clean_rtdetr_training import DenoisingTrainingModel
from clean_rtdetr_training.contracts import DetectionTarget, TrainingOutput
from clean_rtdetr_training.interfaces import TrainingModel

from .teacher import FeatureTeacher


@dataclass(frozen=True)
class DistillationOutput:
    """Pair detection predictions with optional student and teacher feature maps.

    Training retains gradients through the projected student features and
    detaches teacher features. Both maps are absent on the evaluation path.
    """

    detection: TrainingOutput
    student: Float[Tensor, "batch channels height width"] | None = None
    teacher: Float[Tensor, "batch channels teacher_height teacher_width"] | None = None

    @property
    def final(self) -> DetectionOutput:
        """Expose the matching predictions for inspection."""
        return self.detection.final


class DistillationTrainingModel(TrainingModel[DistillationOutput]):
    """Attach a frozen teacher and trainable projector to a DN training model."""

    def __init__(self, student: DenoisingTrainingModel, teacher: FeatureTeacher) -> None:
        """Register all state with ordinary PyTorch ownership."""
        super().__init__()
        if not isinstance(student.model, RTDETR):
            msg = "Distillation requires an RTDETR with an explicit AIFI output"
            raise TypeError(msg)
        self.student = student
        self.teacher = teacher.requires_grad_(requires_grad=False).eval()
        self.projector = nn.Linear(student.model.hidden_dim, teacher.channels)
        self.train(student.training)

    def train(self, mode: bool = True) -> "DistillationTrainingModel":  # noqa: FBT001, FBT002
        """Keep teacher evaluation behavior even when the parent enters training."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        images: Float[Tensor, "batch 3 height width"],
        *,
        targets: Sequence[DetectionTarget] | None = None,
    ) -> DistillationOutput:
        """Return detections and, during training, features for distillation.

        Training requires exactly one student AIFI feature map, projects it to
        the teacher channel count, and returns detached teacher features.
        Evaluation skips the teacher and projector. Detection output retains
        the student's matching and DN predictions for the paired criterion.
        """
        detection = self.student(images, targets=targets, collect_aifi=self.training)
        if not self.training:
            return DistillationOutput(detection)
        features = detection.predictions.after_aifi
        if features is None or len(features.levels) != 1:
            msg = "Student must return exactly one AIFI feature map"
            raise ValueError(msg)
        features.validate()
        feature = features.levels[0]
        projected = self.projector(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.no_grad():
            teacher = self.teacher(images)
        return DistillationOutput(detection, projected, teacher.detach())

    def predict(self, images: Float[Tensor, "batch 3 height width"]) -> DistillationOutput:
        """Run only the detector and restore each submodule's previous mode."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return DistillationOutput(self.student.predict(images))
        finally:
            for module, mode in modes:
                module.training = mode

    def checkpoint_config(self) -> dict[str, object]:
        """Keep teacher provenance and student configuration in resume checks."""
        return {
            "student": self.student.checkpoint_config(),
            "teacher_class": f"{type(self.teacher).__module__}.{type(self.teacher).__qualname__}",
            "teacher": self.teacher.checkpoint_config(),
            "projector": "linear",
        }
