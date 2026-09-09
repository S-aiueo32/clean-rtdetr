# Copyright 2026 Clean RT-DETR contributors
"""Small, locally extracted Penn-Fudan pedestrian detection benchmark."""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from clean_rtdetr.inference import preprocess_images
from clean_rtdetr_training.contracts import DetectionTarget
from clean_rtdetr_training.data.contracts import DetectionSample
from clean_rtdetr_training.interfaces import TrainingModel
from clean_rtdetr_training.models.rtdetr import DenoisingTrainingModel


class PennFudanDetectionDataset(Dataset[DetectionSample]):
    """Read RGB images and derive person boxes from instance masks.

    Labels use zero for person. Images are bilinearly resized to 640 square
    and scaled to [0, 1]. Boxes enclose whole mask pixels (exclusive maxima).
    Sorted filenames are shuffled with a private seed; the last 50 form val.
    This is a project split, not an official benchmark protocol.
    """

    num_classes = 1

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "val", "all"] = "train",
        seed: int = 41,
        val_size: int = 50,
    ) -> None:
        """Load a local PennFudanPed directory without downloading data."""
        if split not in ("train", "val", "all"):
            msg = "split must be 'train', 'val', or 'all'"
            raise ValueError(msg)
        if type(seed) is not int or not 0 <= seed < 2**63:
            msg = "seed must be an integer in [0, 2**63)"
            raise ValueError(msg)
        self.root = Path(root)
        images = sorted((self.root / "PNGImages").glob("*.png"))
        if not images:
            msg = f"No PNG images found under {self.root / 'PNGImages'}"
            raise FileNotFoundError(msg)
        for path in images:
            mask = self.root / "PedMasks" / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(mask)
        if type(val_size) is not int or not 0 < val_size < len(images):
            msg = "val_size must be a positive integer smaller than the dataset"
            raise ValueError(msg)
        order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed)).tolist()
        indices = {"train": order[:-val_size], "val": order[-val_size:], "all": range(len(images))}
        self.image_paths = tuple(images[index] for index in indices[split])

    def __len__(self) -> int:
        """Return the number of images in the selected split."""
        return len(self.image_paths)

    @classmethod
    def from_training_model(
        cls,
        training_model: TrainingModel,
        *,
        root: str | Path,
        split: Literal["train", "val", "all"] = "train",
        seed: int = 41,
        val_size: int = 50,
    ) -> "PennFudanDetectionDataset":
        """Bind the benchmark to a one-class DN-wrapped RTDETR recipe."""
        if not isinstance(training_model, DenoisingTrainingModel):
            msg = "Penn-Fudan model-aware data requires DenoisingTrainingModel"
            raise TypeError(msg)
        if training_model.model.num_classes != cls.num_classes:
            msg = "Penn-Fudan requires a model with one class (person = 0)"
            raise ValueError(msg)
        return cls(root, split=split, seed=seed, val_size=val_size)

    def __getitem__(self, index: int) -> DetectionSample:
        """Return a normalized RGB image and normalized cxcywh person boxes."""
        from PIL import Image  # noqa: PLC0415

        if not 0 <= index < len(self):
            raise IndexError(index)
        path = self.image_paths[index]
        with Image.open(path) as image:
            width, height = image.size
            pixels = preprocess_images([image]).images[0]
        with Image.open(self.root / "PedMasks" / f"{path.stem}_mask.png") as mask:
            if mask.size != (width, height) or mask.mode not in ("L", "P"):
                msg = f"Expected an 8-bit instance mask matching image dimensions: {path.stem}"
                raise ValueError(msg)
            instances = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(
                height, width
            )
        boxes = []
        for instance in instances.unique().tolist():
            if instance == 0:
                continue
            ys, xs = torch.where(instances == instance)
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            boxes.append(
                [
                    (left + right) / (2 * width),
                    (top + bottom) / (2 * height),
                    (right - left) / width,
                    (bottom - top) / height,
                ]
            )
        return pixels, DetectionTarget(
            labels=torch.zeros(len(boxes), dtype=torch.int64),
            boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        )
