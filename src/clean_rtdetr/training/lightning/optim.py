# Copyright 2026 Clean RT-DETR contributors
"""Injectable parameter grouping policy for the pinned RT-DETR AdamW recipe."""

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


def _official_name(name: str) -> str:
    """Map wrapper parameter names to the upstream optimizer's regex domain."""
    if name.startswith(("model.backbone.", "model.encoder.")):
        return name.removeprefix("model.")
    if name == "denoising.class_embed.weight":
        return "decoder.denoising_class_embed.weight"
    for stage in ("projector", "selector", "decoder"):
        prefix = f"model.{stage}."
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            if stage == "decoder" and suffix.startswith("layers."):
                return f"decoder.decoder.{suffix}"
            return f"decoder.{suffix}"
    msg = f"Unrecognized training parameter: {name}"
    raise ValueError(msg)


@dataclass(frozen=True)
class RTDETRParameterGroups:
    """Retain backbone LR and bias/norm decay overrides independently of optimizer type."""

    backbone_lr: float = 1e-5

    def __post_init__(self) -> None:
        """Reject nonpositive or nonfinite learning rates."""
        if not math.isfinite(self.backbone_lr) or self.backbone_lr <= 0:
            msg = "Backbone learning rate must be finite and positive"
            raise ValueError(msg)

    def __call__(self, model: nn.Module) -> list[dict[str, Any]]:
        """Partition every trainable parameter exactly once using upstream names."""
        groups: dict[str, list[Tensor]] = {key: [] for key in ("backbone", "no_decay", "default")}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            official = _official_name(name)
            if official.startswith("backbone."):
                group = "backbone"
            elif re.search(r"^(?=.*(?:encoder|decoder))(?=.*(?:bias|norm.*weight)).*$", official):
                group = "no_decay"
            else:
                group = "default"
            groups[group].append(parameter)
        result: list[dict[str, Any]] = []
        for name, values in groups.items():
            if not values:
                continue
            group_config: dict[str, Any] = {"params": values}
            if name == "backbone":
                group_config["lr"] = self.backbone_lr
            elif name == "no_decay":
                group_config["weight_decay"] = 0.0
            result.append(group_config)
        return result
