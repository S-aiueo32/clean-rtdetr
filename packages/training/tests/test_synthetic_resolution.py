# Copyright 2026 Clean RT-DETR contributors
"""Explicit synthetic data configuration remains independent of detector layout."""

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from clean_rtdetr_training.data import SyntheticDetectionDataset, collate_detection_batch
from clean_rtdetr_training.lightning import DetectionDataModule


@pytest.mark.parametrize("image_size", [32, 65, 128, 256])
def test_explicit_image_size_is_preserved(image_size: int) -> None:
    """The dataset owns pixel generation; model input restrictions belong to its caller."""
    dataset = SyntheticDetectionDataset(image_size=image_size, num_classes=2)
    assert dataset.image_size == image_size
    assert dataset[0][0].shape == (3, image_size, image_size)


@pytest.mark.parametrize("field", ["size", "image_size", "num_classes"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_dataset_dimensions_are_rejected(field: str, value: int) -> None:
    """Reject nonpositive dimensions and booleans without constructing a model."""
    with pytest.raises(ValueError, match="positive integers"):
        SyntheticDetectionDataset(**{field: value})


def test_explicit_class_count_controls_labels() -> None:
    """Generate all configured foreground labels while retaining empty targets."""
    num_classes = 3
    dataset = SyntheticDetectionDataset(size=6, image_size=16, num_classes=num_classes)
    labels = [target.labels.tolist() for _, target in dataset]
    assert labels == [[0], [], [2], [], [1], []]
    assert dataset.num_classes == num_classes


def test_datamodule_defers_independent_dataset_until_setup() -> None:
    """A no-argument factory creates data once and retains the shared loader RNG."""
    calls: list[None] = []

    def dataset_factory() -> SyntheticDetectionDataset:
        calls.append(None)
        return SyntheticDetectionDataset(image_size=32, num_classes=2)

    generator = torch.Generator().manual_seed(41)
    datamodule = DetectionDataModule(
        dataset_factory=dataset_factory,
        dataloader_factory=partial(DataLoader, batch_size=1, collate_fn=collate_detection_batch),
        generator=generator,
    )
    assert calls == []
    datamodule.setup("fit")
    dataset = datamodule.train_dataloader().dataset
    datamodule.setup("fit")
    assert calls == [None]
    loader = datamodule.train_dataloader()
    assert loader.dataset is dataset
    assert loader.generator is generator
