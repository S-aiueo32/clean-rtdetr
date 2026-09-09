# Copyright 2026 Clean RT-DETR contributors
"""Resume validates behavior of injected decoder layers and selector heads."""

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import Tensor, nn

from clean_rtdetr.models.decoder import Decoder, DecoderLayer
from clean_rtdetr.models.selector import BoxHead, QuerySelector
from clean_rtdetr.training.lightning import RTDETRLightningModule
from tests.test_lightning_training import _module

if TYPE_CHECKING:
    from clean_rtdetr.models.rtdetr import BaselineRTDETR


class ScaledLinear(nn.Linear):
    """Keep identical tensor state while exposing a behavior-changing setting."""

    def __init__(self, in_features: int, out_features: int, scale: float) -> None:
        """Construct a linear head with explicit scaling provenance."""
        super().__init__(in_features, out_features)
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002 - match torch.nn.Linear
        """Scale the ordinary linear output without adding tensor state."""
        return super().forward(input) * self.scale

    def checkpoint_config(self) -> dict[str, object]:
        """Record all construction settings that determine the head's behavior."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
            "scale": self.scale,
        }


def _inject(module: RTDETRLightningModule, component: str, setting: float) -> None:
    model = cast("BaselineRTDETR", module.training_model.model)
    if component == "layer":
        config = model.decoder.config
        model.decoder = Decoder(config, layer=DecoderLayer(replace(config, dropout=setting)))
    else:
        config = model.selector.config
        head = ScaledLinear(
            config.hidden_dim, config.num_classes if component == "score" else 4, setting
        )
        model.selector = QuerySelector(
            config,
            score_head=head if component == "score" else None,
            box_head=head if component == "box" else None,
        )


@pytest.mark.parametrize("component", ["layer", "score", "box"])
def test_resume_checks_injected_configuration_with_identical_tensor_shapes(component: str) -> None:
    """Identical injections resume, but changed dropout or head scaling is rejected."""
    original = _module()
    restored = _module()
    _inject(original, component, 0.25)
    _inject(restored, component, 0.25)
    checkpoint = {
        "clean_rtdetr_recipe": original._recipe(),
        "clean_rtdetr_execution": {},
    }
    restored.load_state_dict(original.state_dict())
    restored.on_load_checkpoint(checkpoint)
    _inject(restored, component, 0.5)
    # Weights alone cannot detect the changed behavior.
    restored.load_state_dict(original.state_dict())
    with pytest.raises(ValueError, match="recipe"):
        restored.on_load_checkpoint(checkpoint)


class OpaqueLinear(nn.Linear):
    """A custom head without a declaration of its checkpoint settings."""


class OpaqueBoxHead(BoxHead):
    """A subclass must explicitly own its configuration contract."""


class OpaqueDecoderLayer(DecoderLayer):
    """Inherited baseline provenance cannot describe arbitrary subclass behavior."""


@pytest.mark.parametrize("component", ["layer", "score", "box"])
def test_opaque_injections_run_but_reject_checkpoint_recipes(component: str) -> None:
    """Checkpoint restrictions must not disable ordinary extension forward execution."""
    module = _module()
    model = cast("BaselineRTDETR", module.training_model.model)
    if component == "layer":
        config = model.decoder.config
        model.decoder = Decoder(config, layer=OpaqueDecoderLayer(config))
    else:
        config = model.selector.config
        model.selector = QuerySelector(
            config,
            score_head=OpaqueLinear(config.hidden_dim, config.num_classes)
            if component == "score"
            else None,
            box_head=OpaqueBoxHead(config.hidden_dim) if component == "box" else None,
        )
    module.eval()
    with torch.no_grad():
        assert module(torch.zeros(1, 3, 64, 64)).final.logits.shape == (1, 2, 3)
    with pytest.raises(TypeError, match="explicitly implement checkpoint_config"):
        module._recipe()
