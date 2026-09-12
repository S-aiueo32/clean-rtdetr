# Copyright 2026 Clean RT-DETR contributors
"""Validate or run resumable R50vd training on real COCO images.

``validate`` compares uninterrupted training with an epoch-boundary split and
fresh Lightning resume over two epochs. ``train`` is the matching
epoch-boundary long-job entry point. Neither command measures convergence or detection quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning
import numpy as np
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecision
from PIL import Image
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset

from clean_rtdetr import create_model
from clean_rtdetr_training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    DetectionTarget,
    HungarianMatcher,
    SetCriterion,
)
from clean_rtdetr_training.lightning import (
    RTDETREMA,
    DetectionDataModule,
    EpochBoundaryState,
    RTDETRLightningModule,
    RTDETRParameterGroups,
)

# This checkout-only acceptance tool reuses the pinned exact-resume test probe.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/training/tests"))
_ResumeProbe = importlib.import_module("test_gpu_lightning")._ResumeProbe  # noqa: SLF001 - reuse exact restoration acceptance

if TYPE_CHECKING:
    from collections.abc import Sequence

IMAGE_SIZE = 640
NUM_CLASSES = 80
MODEL_NAME = "rtdetr_r50vd"
EMA_KEY = "RTDETREMA"
RNG_KEY = "EpochBoundaryState"
DATA_KEY = "COCODatasetState"
MINIMUM_UPDATES = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class COCOSample:
    """Store one selected image and its validated absolute boxes."""

    image_id: int
    path: Path
    width: int
    height: int
    annotations: tuple[tuple[int, float, float, float, float], ...]


class COCO640Dataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Read validated real COCO boxes and stretch images to 640 by 640."""

    def __init__(self, image_root: Path, annotation: Path, limit: int | None) -> None:  # noqa: C901
        """Initialize the validated data or callback state."""
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        category_ids = sorted(int(item["id"]) for item in payload["categories"])
        if len(category_ids) != NUM_CLASSES or len(set(category_ids)) != NUM_CLASSES:
            msg = f"Expected {NUM_CLASSES} unique COCO categories, found {len(category_ids)}"
            raise ValueError(msg)
        labels = {category_id: label for label, category_id in enumerate(category_ids)}
        images = {int(item["id"]): item for item in payload["images"]}
        grouped: dict[int, list[tuple[int, float, float, float, float]]] = {
            image_id: [] for image_id in images
        }
        for item in payload["annotations"]:
            if int(item.get("iscrowd", 0)) != 0:
                continue
            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            if image_id not in images or category_id not in labels:
                msg_0 = f"Annotation references unknown image/category: {item.get('id')}"
                raise ValueError(msg_0)
            x, y, width, height = (float(value) for value in item["bbox"])
            image = images[image_id]
            image_width, image_height = int(image["width"]), int(image["height"])
            if not all(np.isfinite((x, y, width, height))) or width <= 0 or height <= 0:
                continue
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(image_width), x + width), min(float(image_height), y + height)
            if x1 > x0 and y1 > y0:
                grouped[image_id].append((labels[category_id], x0, y0, x1 - x0, y1 - y0))
        selected = [image_id for image_id in sorted(images) if grouped[image_id]]
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            msg_0 = "No non-crowd COCO samples with valid boxes were selected"
            raise ValueError(msg_0)
        self.samples = tuple(
            COCOSample(
                image_id=image_id,
                path=image_root / str(images[image_id]["file_name"]),
                width=int(images[image_id]["width"]),
                height=int(images[image_id]["height"]),
                annotations=tuple(grouped[image_id]),
            )
            for image_id in selected
        )
        missing = [str(sample.path) for sample in self.samples if not sample.path.is_file()]
        if missing:
            msg_0 = f"Missing selected COCO image: {missing[0]}"
            raise FileNotFoundError(msg_0)
        self.identity = {
            "version": 1,
            "annotation_sha256": _sha256(annotation),
            "samples": [
                {
                    "image_id": sample.image_id,
                    "image_sha256": _sha256(sample.path),
                    "width": sample.width,
                    "height": sample.height,
                    "annotations": sample.annotations,
                }
                for sample in self.samples
            ],
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
        }

    def __len__(self) -> int:
        """Return the number of selected real images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        """Load RGB pixels and normalize clipped COCO targets."""
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            if source.size != (sample.width, sample.height):
                msg = f"Image dimensions changed for {sample.path}"
                raise ValueError(msg)
            resized = source.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            array = np.array(resized, dtype=np.float32, copy=True) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        labels = torch.tensor([item[0] for item in sample.annotations], dtype=torch.int64)
        boxes = torch.tensor(
            [
                (
                    (x + width / 2) / sample.width,
                    (y + height / 2) / sample.height,
                    width / sample.width,
                    height / sample.height,
                )
                for _, x, y, width, height in sample.annotations
            ],
            dtype=torch.float32,
        )
        if not bool(((boxes >= 0) & (boxes <= 1)).all()):
            msg = f"Normalized boxes are outside [0, 1] for {sample.path}"
            raise ValueError(msg)
        return image, DetectionTarget(labels, boxes)


def _collate(
    samples: Sequence[tuple[Tensor, DetectionTarget]],
) -> tuple[Tensor, list[DetectionTarget]]:
    return torch.stack([image for image, _ in samples]), [target for _, target in samples]


class COCODatasetState(Callback):
    """Reject resume when selected images, annotations, or settings differ."""

    @property
    def state_key(self) -> str:
        """Identify the dataset callback in Lightning checkpoints."""
        return DATA_KEY

    def __init__(self, identity: dict[str, Any], settings: dict[str, Any]) -> None:
        """Initialize the validated data or callback state."""
        self.identity = identity
        self.settings = settings

    def state_dict(self) -> dict[str, Any]:
        """Record data identity and execution settings."""
        return {"identity": self.identity, "settings": self.settings}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Reject a checkpoint with different data or settings."""
        if state_dict != self.state_dict():
            msg = "COCO dataset identity or training settings do not match checkpoint"
            raise ValueError(msg)


class RunEvidence(Callback):
    """Record finite losses, successful optimizer updates, and AMP scale changes."""

    def __init__(self) -> None:
        """Initialize the validated data or callback state."""
        self.losses: list[float] = []
        self.scales: list[float] = []
        self.optimizer_steps = 0
        self._handles: list[Any] = []

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Count executed optimizer steps after Lightning restores state."""
        del pl_module

        def after_step(optimizer: Any, args: Any, kwargs: Any) -> None:  # noqa: ANN401
            del args, kwargs
            gradients = [
                p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None
            ]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                msg = "Executed optimizer step has nonfinite gradients"
                raise FloatingPointError(msg)
            if not any(torch.count_nonzero(g) for g in gradients):
                msg = "Executed optimizer step has no nonzero gradients"
                raise FloatingPointError(msg)
            if not all(
                torch.isfinite(p).all() for g in optimizer.param_groups for p in g["params"]
            ):
                msg = "Optimizer produced nonfinite parameters"
                raise FloatingPointError(msg)
            self.optimizer_steps += 1

        self._handles = [
            optimizer.register_step_post_hook(after_step) for optimizer in trainer.optimizers
        ]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Record finite batch loss and current AMP scale."""
        del pl_module, batch, batch_idx
        loss = cast("Tensor", outputs["loss"] if isinstance(outputs, dict) else outputs)
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            msg = "Observed a non-finite training loss"
            raise FloatingPointError(msg)
        self.losses.append(value)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.scales.append(float(scaler.get_scale()))

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Remove optimizer hooks after training finishes."""
        del trainer, pl_module, stage
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - exercised by EpochBoundaryState.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _settings(args: argparse.Namespace, dataset: COCO640Dataset) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "dataset_samples": len(dataset),
        "precision": args.precision,
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": 1e-4,
        },
        "scheduler": {"name": "MultiStepLR", "milestones": list(args.milestones), "gamma": 0.1},
        "gradient_clip_val": 0.1,
        "num_denoising": args.num_denoising,
    }


def _module(args: argparse.Namespace) -> RTDETRLightningModule[Any]:
    detector = create_model(MODEL_NAME, num_classes=NUM_CLASSES)
    training_model = DenoisingTrainingModel(
        detector,
        DenoisingQueryGenerator(
            NUM_CLASSES, detector.hidden_dim, DenoisingConfig(num_denoising=args.num_denoising)
        ),
    )
    return RTDETRLightningModule(
        training_model,
        SetCriterion(matcher=HungarianMatcher()),
        partial(torch.optim.AdamW, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999)),
        partial(torch.optim.lr_scheduler.MultiStepLR, milestones=args.milestones, gamma=0.1),
        RTDETRParameterGroups(backbone_lr=args.backbone_lr),
        required_callback_keys=(EMA_KEY, RNG_KEY, DATA_KEY),
    )


def _objects(
    args: argparse.Namespace,
    dataset: COCO640Dataset,
    root: Path,
    epochs: int,
    *,
    save: bool,
) -> tuple[RTDETRLightningModule[Any], DetectionDataModule, Trainer, RTDETREMA, RunEvidence]:
    generator = torch.Generator().manual_seed(args.seed)
    data = DetectionDataModule(
        lambda: dataset,
        partial(
            DataLoader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
            drop_last=False,
        ),
        generator,
    )
    ema, evidence = RTDETREMA(), RunEvidence()
    callbacks: list[Callback] = [
        EpochBoundaryState(generator),
        ema,
        COCODatasetState(dataset.identity, _settings(args, dataset)),
        evidence,
    ]
    if save:
        callbacks.append(
            ModelCheckpoint(
                dirpath=root / "checkpoints",
                filename="epoch-{epoch:04d}",
                save_last=True,
                save_on_train_epoch_end=True,
                every_n_epochs=1,
                save_top_k=1,
            )
        )
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true" if args.precision == "32-true" else None,
        plugins=[MixedPrecision("16-mixed", "cuda", torch.amp.GradScaler("cuda", init_scale=128))]
        if args.precision == "16-mixed"
        else [],
        max_epochs=epochs,
        default_root_dir=root,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=save,
        num_sanity_val_steps=0,
        deterministic="warn",
        gradient_clip_val=0.1,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
    )
    return _module(args), data, trainer, ema, evidence


def _cpu_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _max_difference(left: dict[str, Tensor], right: dict[str, Tensor]) -> float:
    return max(
        float((value.float() - right[name].float()).abs().max())
        for name, value in left.items()
        if value.is_floating_point()
    )


def _fit(  # noqa: PLR0913
    args: argparse.Namespace,
    dataset: COCO640Dataset,
    root: Path,
    epochs: int,
    initial: dict[str, Tensor],
    *,
    checkpoint: Path | None = None,
    save: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    _seed(args.seed)
    module, data, trainer, ema, evidence = _objects(args, dataset, root, epochs, save=save)
    module.load_state_dict(initial)
    _seed(args.seed)
    probe = None
    if checkpoint is not None:
        probe = _ResumeProbe(torch.load(checkpoint, map_location="cpu", weights_only=False))
        cast("Any", trainer).callbacks.append(probe)
    trainer.fit(module, datamodule=data, ckpt_path=checkpoint, weights_only=False)
    if probe is not None and not probe.checked:
        msg = "Checkpoint exact restoration was not checked"
        raise RuntimeError(msg)
    if not evidence.losses or evidence.optimizer_steps < 1:
        msg = "Training completed without a successful optimizer update"
        raise RuntimeError(msg)
    if ema.module is None or ema.updates < 1:
        msg = "EMA did not observe a successful optimizer update"
        raise RuntimeError(msg)
    model_state, ema_state = _cpu_state(module), _cpu_state(ema.module)
    result = {
        "global_step": trainer.global_step,
        "successful_optimizer_steps_this_fit": evidence.optimizer_steps,
        "ema_updates_total": ema.updates,
        "finite_losses_this_fit": len(evidence.losses),
        "loss_first": evidence.losses[0],
        "loss_last": evidence.losses[-1],
        "amp_scales": evidence.scales,
        "exact_restoration_checked": probe is not None and probe.checked,
        "parameters_changed": any(
            not torch.equal(initial[name], value.detach().cpu())
            for name, value in module.named_parameters()
        ),
        "scheduler": trainer.lr_scheduler_configs[0].scheduler.state_dict(),
        "precision_state": trainer.precision_plugin.state_dict(),
        "_optimizer_state": move_data_to_device(trainer.optimizers[0].state_dict(), "cpu"),
    }
    del module, data, trainer, ema
    gc.collect()
    torch.cuda.empty_cache()
    return model_state, ema_state, result


def validate(args: argparse.Namespace, dataset: COCO640Dataset) -> dict[str, Any]:  # noqa: C901
    """Compare uninterrupted and resumed training with exact restoration checks."""
    _seed(args.seed + 1)
    initial = _cpu_state(_module(args))
    total_epochs, split_epochs = 2, 1
    full_model, full_ema, full = _fit(
        args, dataset, args.output_dir / "uninterrupted", total_epochs, initial
    )
    _, _, first = _fit(args, dataset, args.output_dir / "split", split_epochs, initial, save=True)
    checkpoint = args.output_dir / "split" / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        msg = "Lightning did not write the epoch-boundary checkpoint"
        raise FileNotFoundError(msg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "callbacks",
        "clean_rtdetr_recipe",
        "clean_rtdetr_execution",
        "clean_rtdetr_required_callbacks",
    }
    if missing := required - saved.keys():
        msg = f"Checkpoint is missing required state: {sorted(missing)}"
        raise ValueError(msg)
    for key in (EMA_KEY, RNG_KEY, DATA_KEY):
        if key not in saved["callbacks"]:
            msg = f"Checkpoint is missing callback state: {key}"
            raise ValueError(msg)
    resumed_model, resumed_ema, resumed = _fit(
        args, dataset, args.output_dir / "resumed", total_epochs, initial, checkpoint=checkpoint
    )
    diagnostic = {
        "uninterrupted": {k: v for k, v in full.items() if not k.startswith("_")},
        "split": {k: v for k, v in first.items() if not k.startswith("_")},
        "resumed": {k: v for k, v in resumed.items() if not k.startswith("_")},
        "max_model_abs_diff": _max_difference(full_model, resumed_model),
        "max_ema_abs_diff": _max_difference(full_ema, resumed_ema),
    }
    (args.output_dir / "continuation-diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )
    tolerance = args.tolerance
    if (
        full["ema_updates_total"] < MINIMUM_UPDATES
        or not full["parameters_changed"]
        or not resumed["parameters_changed"]
    ):
        msg = "Validation requires at least three real parameter updates"
        raise RuntimeError(msg)
    torch.testing.assert_close(
        full.pop("_optimizer_state"),
        resumed.pop("_optimizer_state"),
        rtol=tolerance,
        atol=tolerance,
    )
    first.pop("_optimizer_state")
    if (
        full["scheduler"] != resumed["scheduler"]
        or full["precision_state"] != resumed["precision_state"]
    ):
        msg = "Scheduler/scaler continuation differs"
        raise RuntimeError(msg)
    model_diff = _max_difference(full_model, resumed_model)
    ema_diff = _max_difference(full_ema, resumed_ema)
    for name, value in full_model.items():
        torch.testing.assert_close(value, resumed_model[name], rtol=tolerance, atol=tolerance)
    for name, value in full_ema.items():
        torch.testing.assert_close(value, resumed_ema[name], rtol=tolerance, atol=tolerance)
    if (
        full["global_step"] != resumed["global_step"]
        or full["ema_updates_total"] != resumed["ema_updates_total"]
    ):
        msg = "Resumed optimizer/EMA progress differs from uninterrupted training"
        raise AssertionError(msg)
    return {
        "status": "passed",
        "scope": "short real-image execution and epoch-boundary resume; no convergence claim",
        "dataset_identity": dataset.identity,
        "settings": _settings(args, dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_required_state": sorted(required),
        "first_epoch": first,
        "uninterrupted": full,
        "resumed": resumed,
        "continuation_tolerance": tolerance,
        "max_model_abs_diff": model_diff,
        "max_ema_abs_diff": ema_diff,
    }


def train(args: argparse.Namespace, dataset: COCO640Dataset) -> dict[str, Any]:
    """Run requested epochs and retain resumable Lightning checkpoints."""
    _seed(args.seed + 1)
    initial = _cpu_state(_module(args))
    _, _, result = _fit(
        args, dataset, args.output_dir, args.epochs, initial, checkpoint=args.resume, save=True
    )
    result.pop("_optimizer_state")
    return {
        "status": "completed_requested_epochs",
        "scope": "resumable training execution; no convergence claim",
        "settings": _settings(args, dataset),
        "dataset_identity": dataset.identity,
        "resume": None if args.resume is None else str(args.resume),
        **result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "train"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--precision", choices=("16-mixed", "32-true"), default="16-mixed")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--milestones", type=int, nargs="+", default=[1000])
    parser.add_argument("--num-denoising", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    return parser


def main() -> None:
    """Parse the bounded validation or resumable training command."""
    parser = _parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("full R50vd training requires CUDA")
    if args.batch_size < 1 or args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size, epochs, and optional limit must be positive")
    if args.command == "validate" and args.resume is not None:
        parser.error("validate creates and resumes its own checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = COCO640Dataset(args.image_root, args.annotation, args.limit)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        result = validate(args, dataset) if args.command == "validate" else train(args, dataset)
    report = args.output_dir / "training-report.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
