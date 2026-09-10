# Copyright 2026 Clean RT-DETR contributors
"""Hybrid encoder adapted from lyuwenyu/RT-DETR (Apache-2.0).

Changed: typed feature contracts, explicit AIFI output capture, dynamic positional
encoding, and independent composition. Deploy fusion remains outside this module.
"""

import copy
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from clean_rtdetr.contracts import FeatureSpec, HybridEncoderOutput, MultiScaleFeatures
from clean_rtdetr.models.interfaces import Encoder


@dataclass(frozen=True)
class HybridEncoderConfig:
    """Configuration for the three-level RT-DETR hybrid encoder."""

    hidden_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 1024
    num_encoder_layers: int = 1
    dropout: float = 0.0
    pe_temperature: float = 10000.0
    expansion: float = 1.0
    depth_mult: float = 1.0

    def __post_init__(self) -> None:
        """Reject dimensions and fusion settings unsupported by this AIFI boundary."""
        if min(self.hidden_dim, self.num_heads, self.feedforward_dim, self.num_encoder_layers) <= 0:
            msg = "Encoder dimensions and layer count must be positive"
            raise ValueError(msg)
        if self.hidden_dim % self.num_heads or self.hidden_dim % 4:
            msg = "hidden_dim must be divisible by num_heads and four"
            raise ValueError(msg)
        if not 0 <= self.dropout < 1:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.pe_temperature, self.expansion, self.depth_mult)
        ):
            msg = "Position temperature, expansion, and depth must be finite and positive"
            raise ValueError(msg)
        if int(self.hidden_dim * self.expansion) < 1 or round(3 * self.depth_mult) < 1:
            msg = "Fusion expansion and depth must produce at least one channel and block"
            raise ValueError(msg)


class ConvNormAct(nn.Module):
    """The convolution, batch norm, and activation used by fusion blocks."""

    def __init__(
        self, channels_in: int, channels_out: int, kernel: int, stride: int = 1, *, act: bool = True
    ) -> None:
        """Build an unfused convolution/BN pair with optional SiLU."""
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel, stride, kernel // 2, bias=False)
        self.norm = nn.BatchNorm2d(channels_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        """Apply convolution, normalization, and activation in upstream order."""
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    """A training-time RepVGG block (the reference's unfused form)."""

    def __init__(self, channels: int) -> None:
        """Build the parallel 3x3 and 1x1 branches."""
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, 3, act=False)
        self.conv2 = ConvNormAct(channels, channels, 1, act=False)
        self.act = nn.SiLU()

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        """Sum the two normalized convolutions before activation."""
        return self.act(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    """Cross-stage partial fusion layer with reference-compatible submodule names."""

    def __init__(self, channels_in: int, channels_out: int, depth: int, expansion: float) -> None:
        """Construct parallel partial branches and the optional output projection."""
        super().__init__()
        hidden = int(channels_out * expansion)
        self.conv1 = ConvNormAct(channels_in, hidden, 1)
        self.conv2 = ConvNormAct(channels_in, hidden, 1)
        self.bottlenecks = nn.Sequential(*(RepVggBlock(hidden) for _ in range(depth)))
        self.conv3 = (
            ConvNormAct(hidden, channels_out, 1) if hidden != channels_out else nn.Identity()
        )

    def forward(
        self, x: Float[Tensor, "batch channels _height _width"]
    ) -> Float[Tensor, "batch channels_out _height_out _width_out"]:
        """Fuse the transformed and direct partial branches."""
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    """Batch-first post-norm transformer layer with the upstream state-dict layout."""

    def __init__(self, dim: int, heads: int, feedforward: int, dropout: float) -> None:
        """Construct the official post-norm attention and GELU feed-forward layers."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: Float[Tensor, "batch sequence channels"],
        pos_embed: Float[Tensor, "_batch sequence channels"],
    ) -> Float[Tensor, "batch sequence channels"]:
        """Add position only to attention queries and keys, retaining unmodified values.

        Both residual paths use post-normalization. Position affects attention
        scores but is not added to the values carried through the residual.
        """
        query = src + pos_embed
        attended = self.self_attn(query, query, src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(attended))
        return self.norm2(
            src + self.dropout2(self.linear2(self.dropout(functional.gelu(self.linear1(src)))))
        )


class TransformerEncoder(nn.Module):
    """Apply independently owned AIFI transformer layers in sequence."""

    def __init__(self, layer: TransformerEncoderLayer, count: int) -> None:
        """Clone the upstream-initialized prototype once per layer."""
        super().__init__()

        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(count))

    def forward(
        self,
        src: Float[Tensor, "batch sequence channels"],
        pos_embed: Float[Tensor, "_batch sequence channels"],
    ) -> Float[Tensor, "batch sequence channels"]:
        """Propagate features through each position-aware layer."""
        for layer in self.layers:
            src = layer(src, pos_embed)
        return src


class HybridEncoder(Encoder):
    """Project, self-attend, and fuse backbone feature levels."""

    def __init__(
        self,
        specs: Sequence[FeatureSpec],
        config: HybridEncoderConfig | None = None,
        *,
        input_proj: Sequence[nn.Module] | None = None,
    ) -> None:
        """Derive dimensions and register optional input projection modules."""
        super().__init__()
        config = HybridEncoderConfig() if config is None else config
        if len(specs) != 3:  # noqa: PLR2004 - R50vd exposes three pyramid levels
            msg = "HybridEncoder expects exactly three ordered feature specs"
            raise ValueError(msg)
        if any(b.stride != 2 * a.stride for a, b in pairwise(specs)):
            msg = "HybridEncoder feature strides must double at every level"
            raise ValueError(msg)
        self.specs = tuple(specs)
        self.config = config
        self.hidden_dim = config.hidden_dim
        if input_proj is None:
            input_proj = [
                nn.Sequential(
                    nn.Conv2d(spec.channels, config.hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(config.hidden_dim),
                )
                for spec in specs
            ]
        if len(input_proj) != len(specs):
            msg = "Input projections must match the encoder feature levels"
            raise ValueError(msg)
        self.input_proj = nn.ModuleList(input_proj)
        layer = TransformerEncoderLayer(
            config.hidden_dim, config.num_heads, config.feedforward_dim, config.dropout
        )
        self.encoder = nn.ModuleList([TransformerEncoder(layer, config.num_encoder_layers)])
        depth = round(3 * config.depth_mult)
        self.lateral_convs = nn.ModuleList(
            ConvNormAct(config.hidden_dim, config.hidden_dim, 1) for _ in range(2)
        )
        self.fpn_blocks = nn.ModuleList(
            CSPRepLayer(config.hidden_dim * 2, config.hidden_dim, depth, config.expansion)
            for _ in range(2)
        )
        self.downsample_convs = nn.ModuleList(
            ConvNormAct(config.hidden_dim, config.hidden_dim, 3, 2) for _ in range(2)
        )
        self.pan_blocks = nn.ModuleList(
            CSPRepLayer(config.hidden_dim * 2, config.hidden_dim, depth, config.expansion)
            for _ in range(2)
        )

    def aifi_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate existing AIFI parameters, excluding input projections and CCFF.

        Parameters retain their identity and registration, so gradient inspection
        and optimizer grouping do not depend on internal child module names.
        """
        return self.encoder.parameters()

    @property
    def output_specs(self) -> tuple[FeatureSpec, ...]:
        """Describe the common channel dimension and unchanged pyramid strides."""
        return tuple(FeatureSpec(self.hidden_dim, s.stride) for s in self.specs)

    def _position_embedding(
        self, height: int, width: int, *, device: torch.device
    ) -> Float[Tensor, "1 sequence channels"]:
        """Preserve upstream FP32 construction and width-major sinusoidal ordering."""
        dim = self.hidden_dim // 4
        omega = torch.arange(dim, dtype=torch.float32) / dim
        omega = 1.0 / (self.config.pe_temperature**omega)
        # The reference deliberately constructs a width-major meshgrid; retain
        # that ordering even though feature flattening itself is row-major.
        x, y = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing="ij",
        )
        x = x.flatten()[:, None] @ omega[None]
        y = y.flatten()[:, None] @ omega[None]
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).unsqueeze(0).to(device)

    def forward(
        self, features: MultiScaleFeatures, *, collect_aifi: bool = False
    ) -> HybridEncoderOutput:
        """Project, apply AIFI at the highest stride, then fuse across scales.

        RT-DETR, Eq. (1) and Fig. 5 (https://arxiv.org/html/2304.08069v3),
        separate within-scale attention from cross-scale convolutional fusion::

            P3 ──────────────┐
            P4 ──────────────┼→ top-down FPN → bottom-up PAN → outputs
            P5 → AIFI → F5 ──┘

        P3/P4/P5 denote the fine-to-coarse levels, all with ``hidden_dim``
        channels after projection. Output spatial sizes and strides are preserved.
        ``collect_aifi`` exposes F5 before fusion, without detaching its graph.
        """
        self._validate_features(features)
        projected = [
            proj(level) for proj, level in zip(self.input_proj, features.levels, strict=True)
        ]
        projected[2] = self._apply_aifi(projected[2])
        after = (
            MultiScaleFeatures((projected[2],), (self.output_specs[2],)) if collect_aifi else None
        )
        outputs = self._fuse_top_down(projected)
        outputs = self._fuse_bottom_up(outputs)
        return HybridEncoderOutput(MultiScaleFeatures(tuple(outputs), self.output_specs), after)

    def _validate_features(self, features: MultiScaleFeatures) -> None:
        """Check feature specs and spatial ratios before projection and fusion."""
        features.validate_structure()
        if features.specs != self.specs:
            msg = "Input FeatureSpec differs from the encoder's construction spec"
            raise ValueError(msg)
        self._validate_spatial_ratios(features)

    def _validate_spatial_ratios(self, features: MultiScaleFeatures) -> None:
        """Check heights, then widths, against the strides required by pyramid fusion."""
        heights = [level.shape[-2] for level in features.levels]
        widths = [level.shape[-1] for level in features.levels]
        if any(
            h * spec.stride != heights[0] * self.specs[0].stride
            for h, spec in zip(heights, self.specs, strict=True)
        ):
            msg = "Feature heights must follow the declared stride ratios"
            raise ValueError(msg)
        if any(
            w * spec.stride != widths[0] * self.specs[0].stride
            for w, spec in zip(widths, self.specs, strict=True)
        ):
            msg = "Feature widths must follow the declared stride ratios"
            raise ValueError(msg)

    def _apply_aifi(
        self, feature: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch channels height width"]:
        """Apply position-aware self-attention to the highest-stride feature.

        This is the AIFI/reshape portion of RT-DETR Eq. (1), cited in ``forward``::

            [B, C, H5, W5] -> [B, H5*W5, C] -> encoder -> [B, C, H5, W5]

        Tokens use row-major flattening. The positional embedding deliberately
        retains the pinned implementation's width-major sinusoidal ordering.
        """
        for block in self.encoder:
            x = feature.flatten(2).transpose(1, 2)
            h, w = feature.shape[-2:]
            position = self._position_embedding(h, w, device=x.device).to(dtype=x.dtype)
            x = block(x, position)
            feature = x.transpose(1, 2).reshape_as(feature).contiguous()
        return feature

    def _fuse_top_down(
        self, projected: list[Float[Tensor, "batch channels _height _width"]]
    ) -> list[Float[Tensor, "batch channels _height _width"]]:
        """Upsample coarse features and fuse them with each finer level.

        The top-down half of CCFF (RT-DETR Fig. 5) follows this recurrence::

            lateral = lateral_conv(coarse)
            fine = fpn_block(concat(upsample_2x(lateral), projected_fine))

        Concatenation is along channels, with the upsampled branch first. Each
        lateral replaces its coarse entry so PAN later consumes that projection.
        The returned list is ordered fine to coarse; ``projected`` is not mutated.
        """
        outputs = [projected[-1]]
        for i in range(1, len(projected)):
            idx = len(projected) - 1 - i
            block_idx = i - 1
            high = self.lateral_convs[block_idx](outputs[0])
            outputs[0] = high
            high = functional.interpolate(high, scale_factor=2.0, mode="nearest")
            outputs.insert(0, self.fpn_blocks[block_idx](torch.cat((high, projected[idx]), dim=1)))
        return outputs

    def _fuse_bottom_up(
        self, outputs: list[Float[Tensor, "batch channels _height _width"]]
    ) -> list[Float[Tensor, "batch channels _height _width"]]:
        """Update successive coarse levels with downsampled fused features.

        The bottom-up half of CCFF completes fusion in fine-to-coarse order::

            O3 = top_down[0]
            O4 = pan0(concat(down0(O3), top_down[1]))
            O5 = pan1(concat(down1(O4), top_down[2]))

        Each downsampling convolution halves the spatial dimensions. This method
        updates and returns the supplied list; O5 depends on the newly fused O4.
        """
        for idx in range(len(outputs) - 1):
            outputs[idx + 1] = self.pan_blocks[idx](
                torch.cat((self.downsample_convs[idx](outputs[idx]), outputs[idx + 1]), dim=1)
            )
        return outputs

    def checkpoint_config(self) -> dict[str, object]:
        """Describe encoder configuration and input feature specifications."""
        return {"config": asdict(self.config), "specs": tuple(asdict(spec) for spec in self.specs)}
