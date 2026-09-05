# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for image input conversion."""

import subprocess
import sys
from typing import cast

import pytest
import torch
from PIL import Image

from clean_rtdetr.data import PreprocessedImages, preprocess_images


def test_preprocesses_aspect_ratios_and_preserves_sizes() -> None:
    """Resize mixed aspect ratios while retaining source dimensions."""
    first = Image.new("RGB", (3, 5), (10, 20, 30))
    second = torch.randint(0, 256, (3, 7, 2), dtype=torch.uint8)

    output = preprocess_images([first, second])

    assert isinstance(output, PreprocessedImages)
    assert output.images.shape == (2, 3, 640, 640)
    assert output.images.dtype == torch.float32
    assert output.images.device.type == "cpu"
    assert torch.equal(output.original_sizes_wh, torch.tensor([[3, 5], [2, 7]]))


@pytest.mark.parametrize(("mode", "color"), [("RGBA", (255, 0, 128, 0)), ("L", 128)])
def test_converts_non_rgb_modes_without_exif_transpose(
    mode: str, color: int | tuple[int, int, int, int]
) -> None:
    """Drop alpha and expand grayscale while retaining raw EXIF geometry."""
    image = Image.new(mode, (1, 3), color)
    image.getexif()[274] = 6

    output = preprocess_images([image])

    assert output.images.shape == (1, 3, 640, 640)
    assert output.original_sizes_wh.tolist() == [[1, 3]]


def test_pil_and_tensor_inputs_share_identical_path() -> None:
    """Ensure tensor and equivalent PIL input use identical pixels."""
    tensor = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 5, 7)
    image = Image.frombytes("RGB", (7, 5), bytes(tensor.permute(1, 2, 0).flatten().tolist()))

    from_pil = preprocess_images([image])
    from_tensor = preprocess_images([tensor])

    assert torch.equal(from_pil.images, from_tensor.images)


def test_accepts_noncontiguous_tensor() -> None:
    """Accept valid noncontiguous CHW storage by packing it for Pillow."""
    tensor = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 7, 5).transpose(1, 2)

    output = preprocess_images([tensor])

    assert output.original_sizes_wh.tolist() == [[7, 5]]


@pytest.mark.parametrize(
    "value",
    [[], "not images", [torch.zeros(3, 4, 4, dtype=torch.float32)]],
)
def test_rejects_invalid_inputs(value: object) -> None:
    """Reject empty, non-sequence, and invalid-dtype input."""
    with pytest.raises(ValueError, match=r"images|tensor"):
        preprocess_images(cast("list[torch.Tensor | Image.Image]", value))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(4, 4, dtype=torch.uint8),
        torch.zeros(1, 4, 4, dtype=torch.uint8),
        torch.zeros(3, 4, 4, dtype=torch.int8),
        torch.empty(3, 2, 2, device="meta", dtype=torch.uint8),
        torch.zeros(3, 0, 2, dtype=torch.uint8),
    ],
)
def test_rejects_invalid_tensor_contract(tensor: torch.Tensor) -> None:
    """Reject malformed tensor layouts and dtypes."""
    with pytest.raises(ValueError, match=r"tensor|spatial"):
        preprocess_images([tensor])


def test_rejects_unsupported_image_object() -> None:
    """Reject objects that are neither PIL images nor tensors."""
    with pytest.raises(ValueError, match="each image"):
        preprocess_images(cast("list[torch.Tensor | Image.Image]", [object()]))


def test_imports_without_pillow() -> None:
    """Import package modules without requiring Pillow to be installed."""
    script = """
import builtins

real_import = builtins.__import__
def no_pillow(name, *args, **kwargs):
    if name == "PIL" or name.startswith("PIL."):
        raise ImportError("blocked for test")
    return real_import(name, *args, **kwargs)
builtins.__import__ = no_pillow
import clean_rtdetr
import clean_rtdetr.data
import clean_rtdetr.postprocessing
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
