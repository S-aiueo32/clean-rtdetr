# Copyright 2026 Clean RT-DETR contributors
"""Exercise the installed Penn-Fudan dataset through the recipe configuration."""

from pathlib import Path

import torch
from hydra.utils import instantiate
from PIL import Image, ImageDraw

from clean_rtdetr_training import compose_training_config
from clean_rtdetr_training.data import PennFudanDetectionDataset, SyntheticDetectionDataset


def _dataset(root: Path) -> None:
    """Create a small real PNG/mask directory using the library's file contract."""
    (root / "PNGImages").mkdir(parents=True)
    (root / "PedMasks").mkdir()
    for index in range(3):
        Image.new("RGB", (10, 6), "red").save(root / "PNGImages" / f"person{index}.png")
        mask = Image.new("L", (10, 6))
        ImageDraw.Draw(mask).rectangle((2, 1, 5, 3), fill=1)
        mask.save(root / "PedMasks" / f"person{index}_mask.png")


def test_default_data_is_installed_penn_fudan(tmp_path: Path) -> None:
    """Compose real data without reimplementing its reader or mask conversion."""
    _dataset(tmp_path)
    config = compose_training_config(
        "train",
        [f"data.root={tmp_path}", "data.val_size=1"],
        config_dir=Path(__file__).parents[1] / "configs",
    )
    dataset = instantiate(config.data)()
    assert isinstance(dataset, PennFudanDetectionDataset)
    expected_train_size = 2
    assert len(dataset) == expected_train_size
    image, target = dataset[0]
    assert image.shape == (3, 640, 640)
    torch.testing.assert_close(target.labels, torch.tensor([0]))
    torch.testing.assert_close(target.boxes, torch.tensor([[0.4, 5 / 12, 0.4, 0.5]]))


def test_synthetic_remains_an_explicit_override() -> None:
    """Data-free smoke runs link class counts in config without constructing a model."""
    config = compose_training_config(
        "train",
        ["data=synthetic", "module.training_model.model.num_classes=2"],
        config_dir=Path(__file__).parents[1] / "configs",
    )
    dataset = instantiate(config.data)()
    assert isinstance(dataset, SyntheticDetectionDataset)
    assert dataset.num_classes == 2  # noqa: PLR2004
