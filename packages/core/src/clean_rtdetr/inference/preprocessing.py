# Copyright 2026 Clean RT-DETR contributors
"""Convert image inputs into model-ready tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int64, Shaped, UInt8  # noqa: TC002
from torch import Tensor

if TYPE_CHECKING:
    from PIL import Image

    type ImageInput = Image.Image | UInt8[Tensor, "3 height width"]
else:
    # Keep Pillow optional at runtime; the concrete alias is for type checkers.
    type ImageInput = object

_IMAGE_SIZE = (640, 640)
_RGB_CHANNELS = 3


@dataclass(frozen=True)
class PreprocessedImages:
    """CPU FP32 RGB images and source sizes for postprocessing.

    ``images`` contains values in ``[0,1]``; ``original_sizes_wh`` uses
    ``(width, height)`` for postprocessing.
    """

    images: Float[Tensor, "batch 3 640 640"]
    original_sizes_wh: Int64[Tensor, "batch 2"]


def preprocess_images(images: Sequence[ImageInput]) -> PreprocessedImages:
    """Convert a nonempty sequence of Pillow images or CPU uint8 RGB CHW tensors.

    Every input follows Pillow RGB conversion and bilinear resize to 640x640.
    There is no letterbox, mean/std normalization, or EXIF rotation.
    """
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        msg = "images must be a nonempty sequence"
        raise ValueError(msg)  # noqa: TRY004
    if not images:
        msg = "images must be a nonempty sequence"
        raise ValueError(msg)

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        msg = "preprocess_images requires Pillow; install clean-rtdetr[inference]"
        raise ImportError(msg) from error

    processed: list[UInt8[Tensor, "3 height width"]] = []
    sizes: list[tuple[int, int]] = []
    for image in images:
        pillow_image = _to_rgb_image(image)
        resized = pillow_image.resize(_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        processed.append(_pillow_to_tensor(resized.tobytes(), resized.width, resized.height))
        sizes.append(pillow_image.size)

    return PreprocessedImages(
        images=torch.stack(processed).to(dtype=torch.float32).div(255),
        original_sizes_wh=torch.tensor(sizes, dtype=torch.int64, device="cpu"),
    )


def _to_rgb_image(image: ImageInput) -> Image.Image:
    """Validate one input and convert it to Pillow RGB without resizing."""
    from PIL import Image  # noqa: PLC0415

    if isinstance(image, Tensor):
        _validate_tensor_image(image)
        height, width = (int(image.shape[1]), int(image.shape[2]))
        return Image.frombytes(
            "RGB",
            (width, height),
            bytes(image.permute(1, 2, 0).contiguous().flatten().tolist()),
        )
    if isinstance(image, Image.Image):
        width, height = image.size
        if width <= 0 or height <= 0:
            msg = "images must have positive spatial dimensions"
            raise ValueError(msg)
        return image.convert("RGB")
    msg = "each image must be a PIL image or a CPU uint8 CHW tensor"
    raise ValueError(msg)


def _validate_tensor_image(image: Shaped[Tensor, ...]) -> None:
    if image.device.type != "cpu":
        msg = "tensor images must be on CPU"
        raise ValueError(msg)
    if image.dtype != torch.uint8:
        msg = "tensor images must have dtype torch.uint8"
        raise ValueError(msg)
    if image.ndim != _RGB_CHANNELS or image.shape[0] != _RGB_CHANNELS:
        msg = "tensor images must have shape [3, H, W]"
        raise ValueError(msg)
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        msg = "images must have positive spatial dimensions"
        raise ValueError(msg)


def _pillow_to_tensor(
    image_bytes: bytes, width: int, height: int
) -> UInt8[Tensor, "3 height width"]:
    """Read RGB bytes without introducing a NumPy dependency."""
    return (
        torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        .reshape(height, width, _RGB_CHANNELS)
        .permute(2, 0, 1)
        .contiguous()
    )
