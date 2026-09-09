# Copyright 2026 Clean RT-DETR contributors
"""Resolve synthetic inputs from a constructed model before loading data."""

from functools import partial
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from test_training_interfaces import ToyTrainingModel
from torch.utils.data import DataLoader

from clean_rtdetr_training import DenoisingTrainingModel, compose_training_config
from clean_rtdetr_training.data import SyntheticDetectionDataset, collate_detection_batch
from clean_rtdetr_training.interfaces import TrainingModel
from clean_rtdetr_training.lightning import DetectionDataModule


def _model(num_queries: int = 300) -> DenoisingTrainingModel:
    config = compose_training_config(
        "lightning_training",
        config_dir=Path(__file__).resolve().parents[3] / "configs",
        overrides=[f"module.training_model.model.selector.config.num_queries={num_queries}"],
    )
    with torch.device("meta"):
        return instantiate(config.module.training_model, _convert_="all")


@pytest.mark.parametrize(
    ("num_queries", "expected_size"), [(2, 64), (84, 64), (85, 96), (300, 128), (337, 160)]
)
def test_auto_size_tracks_actual_query_capacity(num_queries: int, expected_size: int) -> None:
    """The resolver crosses stride-aligned size boundaries as query counts grow."""
    dataset = SyntheticDetectionDataset.from_training_model(_model(num_queries))
    assert dataset.image_size == expected_size
    assert dataset[0][0].shape == (3, expected_size, expected_size)


@pytest.mark.parametrize("image_size", [128, 256])
def test_explicit_compatible_image_size_is_preserved(image_size: int) -> None:
    """An explicit training condition is never silently resized."""
    dataset = SyntheticDetectionDataset.from_training_model(_model(), image_size=image_size)
    assert dataset.image_size == image_size


@pytest.mark.parametrize(
    ("image_size", "message"),
    [
        (0, "positive"),
        (-1, "positive"),
        (True, "positive"),
        (32, "multiple"),
        (65, "multiple"),
        (96, "memory positions"),
    ],
)
def test_incompatible_image_size_fails_before_sampling(image_size: int, message: str) -> None:
    """Invalid explicit inputs fail before a training forward or DataLoader iteration."""
    with pytest.raises(ValueError, match=message):
        SyntheticDetectionDataset.from_training_model(_model(), image_size=image_size)


def test_data_class_count_comes_from_constructed_model() -> None:
    """Synthetic labels follow the model without a duplicated class-count setting."""
    num_classes = 2
    config = compose_training_config(
        "lightning_training",
        config_dir=Path(__file__).resolve().parents[3] / "configs",
        overrides=[f"module.training_model.model.selector.config.num_classes={num_classes}"],
    )
    with torch.device("meta"):
        model = instantiate(config.module.training_model, _convert_="all")
    dataset = SyntheticDetectionDataset.from_training_model(model)
    assert dataset.num_classes == model.model.num_classes == num_classes
    assert all(bool((target.labels < num_classes).all()) for _, target in dataset)


def test_default_synthetic_factory_rejects_unsupported_training_model() -> None:
    """Custom model families must supply their own compatible dataset factory."""
    with pytest.raises(TypeError, match="wrapping RTDETR"):
        SyntheticDetectionDataset.from_training_model(ToyTrainingModel())


def test_datamodule_binds_model_and_defers_dataset_until_setup() -> None:
    """Construction binds metadata once and retains the shared DataLoader RNG."""
    model = _model()
    calls: list[TrainingModel] = []

    def dataset_factory(*, training_model: TrainingModel) -> SyntheticDetectionDataset:
        calls.append(training_model)
        return SyntheticDetectionDataset.from_training_model(training_model)

    generator = torch.Generator().manual_seed(41)
    datamodule = DetectionDataModule.from_training_model(
        dataset_factory=dataset_factory,
        dataloader_factory=partial(DataLoader, batch_size=1, collate_fn=collate_detection_batch),
        generator=generator,
        training_model=model,
    )
    assert calls == []
    datamodule.setup("fit")
    dataset = datamodule.train_dataloader().dataset
    datamodule.setup("fit")
    assert calls == [model]
    loader = datamodule.train_dataloader()
    assert loader.dataset is dataset
    assert loader.generator is generator
