# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for the local Penn-Fudan benchmark dataset."""

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from PIL import Image

from clean_rtdetr_training.data import PennFudanDetectionDataset


def _write_dataset(root: Path, count: int = 4) -> None:
    images = root / "PNGImages"
    masks = root / "PedMasks"
    images.mkdir(parents=True)
    masks.mkdir()
    for index in range(count):
        Image.new("RGB", (4, 3), (index, 20, 40)).save(images / f"FUDAN_{index}.png")
        mask = Image.new("L", (4, 3), 0)
        if index == 0:
            mask.putpixel((1, 1), 1)
            mask.putpixel((3, 2), 2)
        elif index == 1:
            mask.putpixel((0, 0), 1)
        masks_path = masks / f"FUDAN_{index}_mask.png"
        mask.save(masks_path)


def test_boxes_use_exclusive_maxima_and_single_pixels(tmp_path: Path) -> None:
    """Convert inclusive mask maxima into correct normalized boxes."""
    _write_dataset(tmp_path, count=2)
    dataset = PennFudanDetectionDataset(tmp_path, split="all", val_size=1)
    image, target = dataset[0]

    assert image.shape == (3, 640, 640)
    assert image.dtype == torch.float32
    assert target.labels.tolist() == [0, 0]
    assert torch.allclose(
        target.boxes,
        torch.tensor([[0.375, 0.5, 0.25, 1 / 3], [0.875, 5 / 6, 0.25, 1 / 3]]),
    )

    _, single = dataset[1]
    assert torch.allclose(single.boxes, torch.tensor([[0.125, 1 / 6, 0.25, 1 / 3]]))


def test_empty_mask_has_empty_target_and_preprocessing_is_normalized(tmp_path: Path) -> None:
    """Return normalized pixels and valid empty targets for empty masks."""
    _write_dataset(tmp_path, count=2)
    dataset = PennFudanDetectionDataset(tmp_path, split="all", val_size=1)
    image, target = dataset[0]
    assert float(image.min()) >= 0
    assert float(image.max()) <= 1
    assert target.boxes.shape == (2, 4)

    mask = Image.new("L", (4, 3), 0)
    mask.save(tmp_path / "PedMasks" / "FUDAN_0_mask.png")
    _, empty = dataset[0]
    assert empty.labels.numel() == 0
    assert empty.boxes.shape == (0, 4)


def test_splits_are_deterministic_disjoint_and_complete_without_global_rng(tmp_path: Path) -> None:
    """Use a private generator for reproducible complete train/validation splits."""
    _write_dataset(tmp_path)
    before = torch.random.get_rng_state()
    train = PennFudanDetectionDataset(tmp_path, split="train", seed=7, val_size=2)
    val = PennFudanDetectionDataset(tmp_path, split="val", seed=7, val_size=2)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert set(train.image_paths).isdisjoint(val.image_paths)
    assert set(train.image_paths) | set(val.image_paths) == set(
        (tmp_path / "PNGImages").glob("*.png")
    )
    assert (
        train.image_paths
        == PennFudanDetectionDataset(tmp_path, split="train", seed=7, val_size=2).image_paths
    )


@pytest.mark.parametrize("split", ["bad", "TRAIN"])
def test_invalid_split_is_rejected(tmp_path: Path, split: str) -> None:
    """Reject unsupported split names."""
    _write_dataset(tmp_path)
    with pytest.raises((ValueError, TypeError), match="split"):
        PennFudanDetectionDataset(tmp_path, split=cast("Literal['train', 'val', 'all']", split))


def test_invalid_root_and_val_size_are_rejected(tmp_path: Path) -> None:
    """Reject missing data and impossible validation sizes."""
    with pytest.raises(FileNotFoundError, match="No PNG images"):
        PennFudanDetectionDataset(tmp_path)
    _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="val_size"):
        PennFudanDetectionDataset(tmp_path, val_size=4)
    (tmp_path / "PedMasks" / "FUDAN_0_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="FUDAN_0_mask"):
        PennFudanDetectionDataset(tmp_path)
