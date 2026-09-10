# Copyright 2026 Clean RT-DETR contributors
"""PResNet variant-d backbones adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: the supported architecture is explicit, outputs use feature contracts,
and checkpoint loading is separate from construction. Freeze semantics are retained.
"""

from collections import OrderedDict
from typing import cast

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.contracts import FeatureSpec, MultiScaleFeatures
from clean_rtdetr.models.interfaces import Backbone

IMAGE_RANK = 4
IMAGE_CHANNELS = 3
MAX_STRIDE = 32
MAX_FREEZE_STAGE = 3
DEFAULT_DEPTH = 50
_STAGE_COUNTS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


class FrozenBatchNorm2d(nn.Module):
    """Batch normalization with fixed affine parameters and running statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Initialize fixed normalization buffers."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(  # noqa: PLR0913, PLR0917 - PyTorch override signature
        self,
        state_dict: dict[str, Shaped[Tensor, "..."]],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,  # noqa: FBT001 - PyTorch override signature
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        """Apply fixed affine normalization."""
        weight = cast("Tensor", self.weight)
        running_var = cast("Tensor", self.running_var)
        bias_parameter = cast("Tensor", self.bias)
        running_mean = cast("Tensor", self.running_mean)
        scale = weight.reshape(1, -1, 1, 1) * (running_var.reshape(1, -1, 1, 1) + self.eps).rsqrt()
        bias = bias_parameter.reshape(1, -1, 1, 1) - running_mean.reshape(1, -1, 1, 1) * scale
        return x * scale + bias


class _ConvNorm(nn.Module):
    def __init__(
        self, ch_in: int, ch_out: int, kernel: int, stride: int, *, act: bool = True
    ) -> None:
        """Build the variant-d convolution and normalization unit."""
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, (kernel - 1) // 2, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        return self.act(self.norm(self.conv(x)))


class _BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in: int, ch_out: int, stride: int, *, shortcut: bool) -> None:
        """Build three convolutions and the variant-d average-pooling shortcut."""
        super().__init__()
        self.branch2a = _ConvNorm(ch_in, ch_out, 1, 1)
        self.branch2b = _ConvNorm(ch_out, ch_out, 3, stride)
        self.branch2c = _ConvNorm(ch_out, ch_out * 4, 1, 1, act=False)
        self.shortcut = shortcut
        if not shortcut:
            if stride == 2:  # noqa: PLR2004 - only downsampling shortcuts use pooling
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", _ConvNorm(ch_in, ch_out * 4, 1, 1, act=False)),
                        ]
                    )
                )
            else:
                self.short = _ConvNorm(ch_in, ch_out * 4, 1, stride, act=False)
        self.act = nn.ReLU()

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in: int, ch_out: int, stride: int, *, shortcut: bool) -> None:
        """Build two convolutions and the variant-d average-pooling shortcut."""
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if stride == 2:  # noqa: PLR2004 - only downsampling shortcuts use pooling
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", _ConvNorm(ch_in, ch_out, 1, 1, act=False)),
                        ]
                    )
                )
            else:
                self.short = _ConvNorm(ch_in, ch_out, 1, stride, act=False)
        self.branch2a = _ConvNorm(ch_in, ch_out, 3, stride)
        self.branch2b = _ConvNorm(ch_out, ch_out, 3, 1, act=False)
        self.act = nn.ReLU()

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class _Stage(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock] | type[_BottleNeck],
        ch_in: int,
        ch_out: int,
        count: int,
        stage_num: int,
    ) -> None:
        """Build a residual stage, downsampling except in the first stage."""
        super().__init__()
        blocks = []
        for i in range(count):
            stride = 2 if i == 0 and stage_num != 2 else 1  # noqa: PLR2004 - upstream res2
            blocks.append(block(ch_in, ch_out, stride, shortcut=i != 0))
            ch_in = ch_out * block.expansion
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        for block in self.blocks:
            x = block(x)
        return x


class PResNet(Backbone):
    """ResNet variant-d backbone returning strides 8, 16 and 32."""

    def __init__(
        self, freeze_at: int = 0, *, freeze_norm: bool = True, depth: int = DEFAULT_DEPTH
    ) -> None:
        """Build PResNet without downloads; freeze the stem and optionally residual stages.

        ``freeze_at=-1`` leaves the stem trainable; nonnegative values freeze the
        stem and that many initial residual stages. Frozen normalization uses
        buffers and remains fixed when the module enters training mode.
        """
        super().__init__()
        if not isinstance(depth, int) or isinstance(depth, bool) or depth not in _STAGE_COUNTS:
            msg = "depth must be one of 18, 34, 50, or 101"
            raise ValueError(msg)
        if (
            not isinstance(freeze_at, int)
            or isinstance(freeze_at, bool)
            or not -1 <= freeze_at <= MAX_FREEZE_STAGE
        ):
            msg = "freeze_at must be an integer between -1 and 3"
            raise ValueError(msg)
        self.freeze_at = freeze_at
        self.freeze_norm = freeze_norm
        self.depth = depth
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1_1", _ConvNorm(3, 32, 3, 2)),
                    ("conv1_2", _ConvNorm(32, 32, 3, 1)),
                    ("conv1_3", _ConvNorm(32, 64, 3, 1)),
                ]
            )
        )
        block = _BottleNeck if depth >= DEFAULT_DEPTH else _BasicBlock
        expansion = block.expansion
        stage_channels = (64, 128, 256, 512)
        stages = []
        ch_in = 64
        for index, (ch_out, count) in enumerate(
            zip(stage_channels, _STAGE_COUNTS[depth], strict=True)
        ):
            stages.append(_Stage(block, ch_in, ch_out, count, index + 2))
            ch_in = ch_out * expansion
        self.res_layers = nn.ModuleList(stages)
        self._specs = tuple(
            FeatureSpec(channels * expansion, stride)
            for channels, stride in zip(stage_channels[1:], (8, 16, 32), strict=True)
        )
        if freeze_at >= 0:
            self._freeze(self.conv1)
            for stage in self.res_layers[:freeze_at]:
                self._freeze(stage)
        if freeze_norm:
            self._freeze_norm(self)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Return the ordered output feature specifications."""
        return self._specs

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_norm(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                replacement = FrozenBatchNorm2d(child.num_features, child.eps)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(module, name, replacement)
            else:
                self._freeze_norm(child)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> MultiScaleFeatures:
        """Extract stride 8, 16 and 32 feature maps from an image batch."""
        input_msg = "Input must be a nonempty floating-point [B, 3, H, W] tensor"
        if (
            x.ndim != IMAGE_RANK
            or x.shape[0] <= 0
            or x.shape[1] != IMAGE_CHANNELS
            or min(x.shape[2:]) <= 0
        ):
            raise ValueError(input_msg)
        if not x.is_floating_point():
            raise ValueError(input_msg)
        if x.shape[2] % MAX_STRIDE or x.shape[3] % MAX_STRIDE:
            msg = "Input height and width must be multiples of 32"
            raise ValueError(msg)
        x = self.conv1(x)
        x = functional.max_pool2d(x, 3, 2, 1)
        outputs = []
        for index, stage in enumerate(self.res_layers):
            x = stage(x)
            if index in (1, 2, 3):
                outputs.append(x)
        return MultiScaleFeatures(tuple(outputs), self.specs)

    def checkpoint_config(self) -> dict[str, object]:
        """Describe backbone freezing and output feature settings."""
        return {
            "depth": self.depth,
            "freeze_at": self.freeze_at,
            "freeze_norm": self.freeze_norm,
            "specs": tuple((spec.channels, spec.stride) for spec in self.specs),
        }
