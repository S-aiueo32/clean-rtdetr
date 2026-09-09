# Copyright 2026 Clean RT-DETR contributors
"""Official RT-DETR model presets and named construction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
from torch import nn

from clean_rtdetr.contracts import DecoderPlan
from clean_rtdetr.models.attention import MultiScaleDeformableAttention
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig, DecoderLayer
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.interfaces import Detector
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.rtdetr import RTDETR, BaselineRTDETR
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig

if TYPE_CHECKING:
    from clean_rtdetr.contracts import FeatureSpec

type ModelBuilder = Callable[[int], Detector]
type ModelFamily = Literal["rtdetr", "rtdetrv2"]

_UPSTREAM_COMMIT = "29320b6fd828f8e0987a71426cf2d961b09dfed7"
_UPSTREAM = f"https://github.com/lyuwenyu/RT-DETR/blob/{_UPSTREAM_COMMIT}"
_WEIGHTS = "https://github.com/lyuwenyu/storage/releases/download"


class _PresetChanges(TypedDict, total=False):
    hidden_dim: int
    encoder_expansion: float
    encoder_feedforward_dim: int
    decoder_layers: int
    default_eval_plan: DecoderPlan
    freeze_at: int
    freeze_norm: bool


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """Describe one named upstream architecture and its checkpoint provenance."""

    name: str
    family: ModelFamily
    config_url: str
    checkpoint_url: str
    backbone_depth: Literal[18, 34, 50, 101]
    hidden_dim: int = 256
    encoder_expansion: float = 1.0
    encoder_feedforward_dim: int = 1024
    decoder_layers: int = 6
    default_eval_plan: DecoderPlan | None = None
    freeze_at: int = 0
    freeze_norm: bool = True


class ModelFactory:
    """Build named detectors from an explicit, immutable preset mapping."""

    def __init__(
        self,
        presets: Mapping[str, ModelPreset],
        builders: Mapping[str, ModelBuilder],
    ) -> None:
        """Copy mappings and require one builder for every preset."""
        copied_presets = dict(presets)
        copied_builders = dict(builders)
        if copied_presets.keys() != copied_builders.keys():
            msg = "Model presets and builders must have identical names"
            raise ValueError(msg)
        for name, preset in copied_presets.items():
            if not name or name != preset.name:
                msg = "Model preset keys must be nonempty and match preset names"
                raise ValueError(msg)
            if not callable(copied_builders[name]):
                msg = f"Model builder must be callable: {name}"
                raise TypeError(msg)
        self._presets = MappingProxyType(copied_presets)
        self._builders = MappingProxyType(copied_builders)

    @property
    def names(self) -> tuple[str, ...]:
        """Return supported model names in declaration order."""
        return tuple(self._presets)

    def preset(self, name: str) -> ModelPreset:
        """Return architecture and official-source metadata for a model."""
        try:
            return self._presets[name]
        except KeyError as error:
            raise self._unknown_model(name) from error

    def create(self, name: str, *, num_classes: int = 80) -> Detector:
        """Build a fresh detector from a named preset."""
        try:
            builder = self._builders[name]
        except KeyError as error:
            raise self._unknown_model(name) from error
        model = builder(num_classes)
        if not isinstance(model, Detector):
            msg = f"Model builder {name!r} returned {type(model).__name__}, expected Detector"
            raise TypeError(msg)
        return model

    def _unknown_model(self, name: str) -> ValueError:
        available = ", ".join(self.names) or "none"
        return ValueError(f"Unknown model {name!r}; available models: {available}")


def _build_model(preset: ModelPreset, num_classes: int) -> BaselineRTDETR:
    """Compose one official preset from the repository's explicit stages."""
    backbone = PResNet(
        depth=preset.backbone_depth,
        freeze_at=preset.freeze_at,
        freeze_norm=preset.freeze_norm,
    )
    encoder_config = HybridEncoderConfig(
        hidden_dim=preset.hidden_dim,
        feedforward_dim=preset.encoder_feedforward_dim,
        expansion=preset.encoder_expansion,
    )
    input_proj = None
    encoder_output = None
    decoder_layer = None
    if preset.family == "rtdetrv2":
        input_proj = _v2_input_projections(backbone.specs, preset.hidden_dim)
        encoder_output = _v2_encoder_output(preset.hidden_dim)
    encoder = HybridEncoder(backbone.specs, encoder_config, input_proj=input_proj)
    projector = MemoryProjector(encoder.output_specs, hidden_dim=preset.hidden_dim)
    selector = QuerySelector(
        QuerySelectorConfig(
            hidden_dim=preset.hidden_dim,
            num_classes=num_classes,
            num_queries=300,
            learned_content=False,
        ),
        encoder_output=encoder_output,
    )
    decoder_config = DecoderConfig(
        hidden_dim=preset.hidden_dim,
        num_classes=num_classes,
        num_layers=preset.decoder_layers,
        num_heads=8,
        num_levels=3,
        num_points=4,
    )
    if preset.family == "rtdetrv2":
        decoder_layer = _v2_decoder_layer(decoder_config)
    decoder = Decoder(
        decoder_config,
        layer=decoder_layer,
        default_eval_plan=preset.default_eval_plan,
    )
    return RTDETR(backbone, encoder, projector, selector, QueryAssembler(), decoder)


def _v2_input_projections(specs: tuple[FeatureSpec, ...], hidden_dim: int) -> list[nn.Sequential]:
    """Build injected projections with official RT-DETRv2 state-dict names."""
    return [
        nn.Sequential(
            OrderedDict(
                (
                    ("conv", nn.Conv2d(spec.channels, hidden_dim, 1, bias=False)),
                    ("norm", nn.BatchNorm2d(hidden_dim)),
                )
            )
        )
        for spec in specs
    ]


def _v2_encoder_output(hidden_dim: int) -> nn.Sequential:
    """Build the injected named v2 encoder-output projection."""
    projection = nn.Linear(hidden_dim, hidden_dim)
    nn.init.xavier_uniform_(projection.weight)
    return nn.Sequential(OrderedDict((("proj", projection), ("norm", nn.LayerNorm(hidden_dim)))))


def _v2_decoder_layer(config: DecoderConfig) -> DecoderLayer:
    """Build one v2 layer prototype for injection into the shared decoder."""
    point_scale = torch.full(
        (config.num_levels * config.num_points,),
        1.0 / config.num_points,
    )
    attention = MultiScaleDeformableAttention(
        config.hidden_dim,
        config.num_heads,
        config.num_levels,
        config.num_points,
        point_scale=point_scale,
    )
    return DecoderLayer(
        config,
        cross_attention=attention,
        residual_stabilizer=nn.Identity(),
    )


def _preset(
    name: str,
    family: ModelFamily,
    config: str,
    checkpoint: str,
    depth: Literal[18, 34, 50, 101],
    **changes: Unpack[_PresetChanges],
) -> ModelPreset:
    """Keep the repetitive official URL and preset declarations compact."""
    package = "rtdetr_pytorch" if family == "rtdetr" else "rtdetrv2_pytorch"
    return ModelPreset(
        name=name,
        family=family,
        config_url=f"{_UPSTREAM}/{package}/configs/{family}/{config}",
        checkpoint_url=f"{_WEIGHTS}/{checkpoint}",
        backbone_depth=depth,
        **changes,
    )


_PRESET_SEQUENCE = (
    _preset(
        "rtdetr_r18vd",
        "rtdetr",
        "rtdetr_r18vd_6x_coco.yml",
        "v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        18,
        encoder_expansion=0.5,
        decoder_layers=3,
        freeze_at=-1,
        freeze_norm=False,
    ),
    _preset(
        "rtdetr_r34vd",
        "rtdetr",
        "rtdetr_r34vd_6x_coco.yml",
        "v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
        34,
        encoder_expansion=0.5,
        decoder_layers=4,
        freeze_at=-1,
        freeze_norm=False,
    ),
    _preset(
        "rtdetr_r50vd_m",
        "rtdetr",
        "rtdetr_r50vd_m_6x_coco.yml",
        "v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
        50,
        encoder_expansion=0.5,
        default_eval_plan=DecoderPlan(num_layers=3, collect_intermediate=False),
    ),
    _preset(
        "rtdetr_r50vd",
        "rtdetr",
        "rtdetr_r50vd_6x_coco.yml",
        "v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
        50,
    ),
    _preset(
        "rtdetr_r101vd",
        "rtdetr",
        "rtdetr_r101vd_6x_coco.yml",
        "v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth",
        101,
        hidden_dim=384,
        encoder_feedforward_dim=2048,
    ),
    _preset(
        "rtdetrv2_r18vd",
        "rtdetrv2",
        "rtdetrv2_r18vd_120e_coco.yml",
        "v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        18,
        encoder_expansion=0.5,
        decoder_layers=3,
        freeze_at=-1,
        freeze_norm=False,
    ),
    _preset(
        "rtdetrv2_r34vd",
        "rtdetrv2",
        "rtdetrv2_r34vd_120e_coco.yml",
        "v0.1/rtdetrv2_r34vd_120e_coco_ema.pth",
        34,
        encoder_expansion=0.5,
        decoder_layers=4,
        freeze_at=-1,
        freeze_norm=False,
    ),
    _preset(
        "rtdetrv2_r50vd_m",
        "rtdetrv2",
        "rtdetrv2_r50vd_m_7x_coco.yml",
        "v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        50,
        encoder_expansion=0.5,
        default_eval_plan=DecoderPlan(num_layers=3, collect_intermediate=False),
    ),
    _preset(
        "rtdetrv2_r50vd",
        "rtdetrv2",
        "rtdetrv2_r50vd_6x_coco.yml",
        "v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
        50,
    ),
    _preset(
        "rtdetrv2_r101vd",
        "rtdetrv2",
        "rtdetrv2_r101vd_6x_coco.yml",
        "v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        101,
        hidden_dim=384,
        encoder_feedforward_dim=2048,
    ),
)
_PRESETS = {preset.name: preset for preset in _PRESET_SEQUENCE}


def _named_builder(name: str, num_classes: int) -> BaselineRTDETR:
    return _build_model(_PRESETS[name], num_classes)


_BUILDERS: dict[str, ModelBuilder] = {name: partial(_named_builder, name) for name in _PRESETS}
_MODEL_FACTORY = ModelFactory(_PRESETS, _BUILDERS)


def list_models() -> tuple[str, ...]:
    """Return every official preset accepted by :func:`create_model`."""
    return _MODEL_FACTORY.names


def get_model_preset(name: str) -> ModelPreset:
    """Return resolved construction and official checkpoint metadata."""
    return _MODEL_FACTORY.preset(name)


def create_model(name: str, *, num_classes: int = 80) -> BaselineRTDETR:
    """Build an uninitialized-weight model selected by its official preset name."""
    model = _MODEL_FACTORY.create(name, num_classes=num_classes)
    if not isinstance(model, RTDETR):
        msg = f"Built-in model {name!r} must produce RTDETR"
        raise TypeError(msg)
    return model


def rtdetr_r18vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETR R18vd architecture without weights."""
    return create_model("rtdetr_r18vd", num_classes=num_classes)


def rtdetr_r34vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETR R34vd architecture without weights."""
    return create_model("rtdetr_r34vd", num_classes=num_classes)


def rtdetr_r50vd_m(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official medium RT-DETR R50vd architecture without weights."""
    return create_model("rtdetr_r50vd_m", num_classes=num_classes)


def rtdetr_r50vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the pinned R50vd reference composition without downloading weights."""
    return create_model("rtdetr_r50vd", num_classes=num_classes)


def rtdetr_r101vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETR R101vd architecture without weights."""
    return create_model("rtdetr_r101vd", num_classes=num_classes)


def rtdetrv2_r18vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETRv2 R18vd architecture without weights."""
    return create_model("rtdetrv2_r18vd", num_classes=num_classes)


def rtdetrv2_r34vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETRv2 R34vd architecture without weights."""
    return create_model("rtdetrv2_r34vd", num_classes=num_classes)


def rtdetrv2_r50vd_m(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official medium RT-DETRv2 R50vd architecture without weights."""
    return create_model("rtdetrv2_r50vd_m", num_classes=num_classes)


def rtdetrv2_r50vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETRv2 R50vd architecture without weights."""
    return create_model("rtdetrv2_r50vd", num_classes=num_classes)


def rtdetrv2_r101vd(*, num_classes: int = 80) -> BaselineRTDETR:
    """Build the official RT-DETRv2 R101vd architecture without weights."""
    return create_model("rtdetrv2_r101vd", num_classes=num_classes)
