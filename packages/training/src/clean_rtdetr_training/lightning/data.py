# Copyright 2026 Clean RT-DETR contributors
"""Injected Lightning data-module boundary for detection training."""

from collections.abc import Callable
from functools import partial
from typing import Protocol, runtime_checkable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from clean_rtdetr_training.data.contracts import DetectionSample
from clean_rtdetr_training.interfaces import TrainingModel


@runtime_checkable
class TrainingDatasetFactory(Protocol):
    """Resolve a dataset using an already constructed training model."""

    def __call__(self, *, training_model: TrainingModel) -> Dataset[DetectionSample]:
        """Build a dataset using model metadata when needed."""
        ...


@runtime_checkable
class DetectionDataLoaderFactory(Protocol):
    """Build a detection loader from a dataset and a shared RNG."""

    def __call__(
        self,
        dataset: Dataset[DetectionSample],
        *,
        generator: torch.Generator,
    ) -> DataLoader[DetectionSample]:
        """Return a loader for the supplied dataset and RNG."""
        ...


class DetectionDataModule(LightningDataModule):
    """Own injected dataset/loader factories while preserving epoch RNG state."""

    def __init__(
        self,
        dataset_factory: Callable[[], Dataset[DetectionSample]],
        dataloader_factory: DetectionDataLoaderFactory,
        generator: torch.Generator,
    ) -> None:
        """Store factories and the application-owned generator without constructing data."""
        super().__init__()
        self.dataset_factory = dataset_factory
        self.dataloader_factory = dataloader_factory
        self.generator = generator
        self._train_dataset: Dataset[DetectionSample] | None = None

    @classmethod
    def from_training_model(
        cls,
        dataset_factory: TrainingDatasetFactory,
        dataloader_factory: DetectionDataLoaderFactory,
        generator: torch.Generator,
        training_model: TrainingModel,
    ) -> "DetectionDataModule":
        """Bind the model to a dataset factory; create the dataset during setup."""
        return cls(
            dataset_factory=partial(dataset_factory, training_model=training_model),
            dataloader_factory=dataloader_factory,
            generator=generator,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create the training dataset once for fit or an unspecified stage."""
        if stage not in (None, "fit"):
            return
        if self._train_dataset is None:
            self._train_dataset = self.dataset_factory()

    def train_dataloader(self) -> DataLoader[DetectionSample]:
        """Build the training loader using the shared epoch RNG."""
        if self._train_dataset is None:
            msg = "DetectionDataModule.setup('fit') must run before train_dataloader"
            raise RuntimeError(msg)
        return self.dataloader_factory(self._train_dataset, generator=self.generator)
