# Copyright 2026 Clean RT-DETR contributors
"""Frozen feature teachers used by the RT-DETRv4 training recipe.

The DINOv3 preprocessing and token reshape follow the official implementation
(RT-DETRs/RT-DETRv4, commit ``55fefaaed7efe2a5f72d0a18fd4e05965e35c292``):
``engine/rtv4/dinov3_teacher.py``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_IMAGE_RANK = 4
_IMAGE_CHANNELS = 3
_FEATURE_STRIDE = 32


class FeatureTeacher(nn.Module, ABC):
    """Interface for a frozen NCHW semantic feature teacher."""

    channels: int

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Return detached ``[batch, channels, height, width]`` features."""

    @abstractmethod
    def checkpoint_config(self) -> dict[str, object]:
        """Return stable, serializable teacher provenance."""


class LocalDINOv3Teacher(FeatureTeacher):
    """Load DINOv3 from a caller-owned local checkout without network access."""

    def __init__(self, repo_dir: str | Path, weights: str | Path) -> None:
        """Load a DINOv3 ViT-B/16 checkpoint from local filesystem paths."""
        super().__init__()
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.weights = Path(weights).expanduser().resolve()
        if not self.repo_dir.is_dir():
            msg = f"DINOv3 repository does not exist: {self.repo_dir}"
            raise FileNotFoundError(msg)
        if not self.weights.is_file():
            msg = f"DINOv3 weights do not exist: {self.weights}"
            raise FileNotFoundError(msg)
        loaded_model = torch.hub.load(
            str(self.repo_dir),
            "dinov3_vitb16",
            source="local",
            weights=str(self.weights),
        )
        embed_dim = getattr(loaded_model, "embed_dim", None)
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            msg = "DINOv3 model must expose a positive integer embed_dim"
            raise ValueError(msg)
        self.model: nn.Module = loaded_model
        self.channels = embed_dim
        self.mean: Tensor
        self.std: Tensor
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.requires_grad_(requires_grad=False)
        self.eval()
        with self.weights.open("rb") as weight_file:
            self._weights_sha256 = hashlib.file_digest(weight_file, "sha256").hexdigest()

    def train(self, mode: bool = True) -> LocalDINOv3Teacher:  # noqa: FBT001, FBT002
        """Keep the external teacher in evaluation mode."""
        super().train(mode)
        super().train(mode=False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Run DINOv3 and reshape patch tokens to a rectangular feature map."""
        if images.ndim != _IMAGE_RANK or images.shape[1] != _IMAGE_CHANNELS:
            msg = "images must have shape [batch, 3, height, width]"
            raise ValueError(msg)
        if not images.is_floating_point() or images.shape[0] <= 0:
            msg = "images must be a non-empty floating-point batch"
            raise ValueError(msg)
        height, width = images.shape[-2:]
        if (
            height <= 0
            or width <= 0
            or height % _FEATURE_STRIDE != 0
            or width % _FEATURE_STRIDE != 0
        ):
            msg = "image height and width must be positive and divisible by 32"
            raise ValueError(msg)
        processed = self.pool((images - self.mean) / self.std)
        with torch.no_grad():
            result: Any = self.model(processed, is_training=True, masks=None)
            tokens = result["x_norm_patchtokens"]
        expected = (height // _FEATURE_STRIDE) * (width // _FEATURE_STRIDE)
        expected_shape = (images.shape[0], expected, self.channels)
        if tuple(tokens.shape) != expected_shape:
            msg = f"DINOv3 returned {tuple(tokens.shape)}; expected {expected_shape}"
            raise ValueError(msg)
        return (
            tokens.detach()
            .transpose(1, 2)
            .reshape(
                images.shape[0], self.channels, height // _FEATURE_STRIDE, width // _FEATURE_STRIDE
            )
        )

    def checkpoint_config(self) -> dict[str, object]:
        """Describe local paths and immutable weight provenance."""
        return {
            "type": type(self).__name__,
            "repo_dir": str(self.repo_dir),
            "weights": str(self.weights),
            "weights_sha256": self._weights_sha256,
            "channels": self.channels,
            "patch_size": 16,
            "input_pool": 2,
        }


class SmokeTeacher(FeatureTeacher):
    """Small random teacher for recipe smoke tests; never used for training claims."""

    def __init__(self, channels: int = 16) -> None:
        """Construct a deterministic-shape random teacher for smoke tests."""
        super().__init__()
        if channels <= 0:
            msg = "channels must be positive"
            raise ValueError(msg)
        self.channels = channels
        self.pool = nn.AvgPool2d(kernel_size=32, stride=32)
        self.projection = nn.Conv2d(3, channels, kernel_size=1)
        self.requires_grad_(requires_grad=False)
        self.eval()

    def train(self, mode: bool = True) -> SmokeTeacher:  # noqa: FBT001, FBT002
        """Keep the smoke teacher in evaluation mode."""
        super().train(mode)
        super().train(mode=False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Return pooled and projected smoke features."""
        if images.ndim != _IMAGE_RANK or images.shape[1] != _IMAGE_CHANNELS:
            msg = "images must have shape [batch, 3, height, width]"
            raise ValueError(msg)
        height, width = images.shape[-2:]
        if (
            height <= 0
            or width <= 0
            or height % _FEATURE_STRIDE != 0
            or width % _FEATURE_STRIDE != 0
        ):
            msg = "image height and width must be positive and divisible by 32"
            raise ValueError(msg)
        return self.projection(self.pool(images)).detach()

    def checkpoint_config(self) -> dict[str, object]:
        """Return the smoke teacher configuration."""
        return {"type": type(self).__name__, "channels": self.channels, "stride": 32}
