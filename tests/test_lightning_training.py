# Copyright 2026 Clean RT-DETR contributors
"""Acceptance checks for epoch-boundary Lightning training and resume."""

from __future__ import annotations

import random
import subprocess
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

lightning = pytest.importorskip("lightning")
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

from clean_rtdetr.training import DetectionTarget, SetCriterion  # noqa: E402
from clean_rtdetr.training.denoising import DenoisingConfig, DenoisingQueryGenerator  # noqa: E402
from clean_rtdetr.training.lightning import (  # noqa: E402
    RTDETREMA,
    EpochBoundaryState,
    RTDETRLightningModule,
    RTDETRParameterGroups,
)
from clean_rtdetr.training.model import DenoisingTrainingModel  # noqa: E402
from tests.test_model import small_model  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

EXPECTED_STEPS = 2
DEFAULT_WEIGHT_DECAY = 1e-4


class _NoisyDetectionDataset(Dataset[tuple[torch.Tensor, DetectionTarget]]):
    """A tiny dataset whose samples consume every RNG restored by the callback."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget]:
        noise = (
            random.random()  # noqa: S311 -- exercise checkpointed Python augmentation RNG.
            + float(np.random.random())  # noqa: NPY002 -- exercise checkpointed global NumPy RNG.
            + float(torch.rand(()))
        )
        image = torch.full((3, 64, 64), 0.01 * (index + 1) + 0.001 * noise)
        if index == 0:
            target = DetectionTarget(
                torch.tensor([0, 2]),
                torch.tensor([[0.25, 0.3, 0.2, 0.25], [0.7, 0.65, 0.15, 0.2]]),
            )
        else:
            target = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
        return image, target


def _collate(
    samples: Sequence[tuple[torch.Tensor, DetectionTarget]],
) -> tuple[torch.Tensor, list[DetectionTarget]]:
    return torch.stack([image for image, _ in samples]), [target for _, target in samples]


def _module(*, freeze_backbone: bool = True, optimizer_lr: float = 1e-4) -> RTDETRLightningModule:
    detector = small_model()
    if freeze_backbone:
        detector.backbone.requires_grad_(requires_grad=False)
    wrapped = DenoisingTrainingModel(
        detector,
        DenoisingQueryGenerator(
            num_classes=3,
            hidden_dim=8,
            config=DenoisingConfig(num_denoising=4),
        ),
    )
    return RTDETRLightningModule(
        wrapped,
        SetCriterion(),
        partial(
            torch.optim.AdamW,
            lr=optimizer_lr,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            betas=(0.9, 0.999),
        ),
        partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[1], gamma=0.5),
        RTDETRParameterGroups(backbone_lr=1e-5),
    )


def _loader(generator: torch.Generator, *, workers: int = 0) -> DataLoader[Any]:
    return DataLoader(
        _NoisyDetectionDataset(),
        batch_size=2,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        collate_fn=_collate,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- callback supports NumPy's global RNG.
    torch.manual_seed(seed)


def _trainer(
    root: Path,
    *,
    epochs: int,
    generator: torch.Generator,
    checkpoint: ModelCheckpoint | None = None,
    ema: RTDETREMA | None = None,
    **kwargs: object,
) -> lightning.Trainer:
    callbacks = [EpochBoundaryState(generator)]
    if checkpoint is not None:
        callbacks.append(checkpoint)
    if ema is not None:
        callbacks.append(ema)
    return lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=epochs,
        default_root_dir=root,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=checkpoint is not None,
        num_sanity_val_steps=0,
        deterministic=True,
        **kwargs,
    )


def _optimizer_state(trainer: lightning.Trainer) -> dict[str, Any]:
    assert len(trainer.optimizers) == 1
    return trainer.optimizers[0].state_dict()


def _scheduler_state(trainer: lightning.Trainer) -> dict[str, Any]:
    assert len(trainer.lr_scheduler_configs) == 1
    return trainer.lr_scheduler_configs[0].scheduler.state_dict()


def _assert_nested_equal(left: Any, right: Any) -> None:  # noqa: ANN401
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _run_resume_acceptance(root: Path) -> None:
    """Resume a fresh Trainer with exact optimizer, EMA, and RNG state."""
    _seed_all(101)
    initial = {name: value.clone() for name, value in _module().state_dict().items()}

    _seed_all(202)
    full_generator = torch.Generator().manual_seed(303)
    full_module = _module()
    full_module.load_state_dict(initial)
    _seed_all(202)
    full_ema = RTDETREMA()
    full_trainer = _trainer(root / "full", epochs=2, generator=full_generator, ema=full_ema)
    full_trainer.fit(full_module, train_dataloaders=_loader(full_generator))
    full_python_state = random.getstate()
    full_numpy_state = np.random.get_state()  # noqa: NPY002
    full_torch_state = torch.random.get_rng_state()

    _seed_all(202)
    first_generator = torch.Generator().manual_seed(303)
    first_module = _module()
    first_module.load_state_dict(initial)
    _seed_all(202)
    checkpoint = ModelCheckpoint(
        dirpath=root / "split",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    first_trainer = _trainer(
        root / "split",
        epochs=1,
        generator=first_generator,
        checkpoint=checkpoint,
        ema=RTDETREMA(),
    )
    first_trainer.fit(first_module, train_dataloaders=_loader(first_generator))
    assert checkpoint.last_model_path

    # Deliberately perturb all state before constructing the fresh resume objects.
    _seed_all(909)
    resumed_generator = torch.Generator().manual_seed(808)
    resumed_module = _module()
    resumed_ema = RTDETREMA()
    resumed_trainer = _trainer(
        root / "resumed", epochs=2, generator=resumed_generator, ema=resumed_ema
    )
    resumed_trainer.fit(
        resumed_module,
        train_dataloaders=_loader(resumed_generator),
        ckpt_path=checkpoint.last_model_path,
    )

    for name, full_value in full_module.state_dict().items():
        torch.testing.assert_close(
            full_value,
            resumed_module.state_dict()[name],
            rtol=0,
            atol=0,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
    _assert_nested_equal(_optimizer_state(full_trainer), _optimizer_state(resumed_trainer))
    _assert_nested_equal(_scheduler_state(full_trainer), _scheduler_state(resumed_trainer))
    assert full_trainer.global_step == resumed_trainer.global_step == EXPECTED_STEPS
    assert full_ema.module is not None
    assert resumed_ema.module is not None
    _assert_nested_equal(full_ema.module.state_dict(), resumed_ema.module.state_dict())
    assert full_ema.updates == resumed_ema.updates
    assert torch.equal(full_generator.get_state(), resumed_generator.get_state())
    assert random.getstate() == full_python_state
    numpy_state = np.random.get_state()  # noqa: NPY002
    assert numpy_state[0] == full_numpy_state[0]
    assert np.array_equal(numpy_state[1], full_numpy_state[1])
    assert numpy_state[2:] == full_numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), full_torch_state)

    changed_generator = torch.Generator().manual_seed(303)
    changed_module = _module(optimizer_lr=2e-4)
    changed_trainer = _trainer(
        root / "changed", epochs=2, generator=changed_generator, ema=RTDETREMA()
    )
    with pytest.raises(ValueError, match="recipe"):
        changed_trainer.fit(
            changed_module,
            train_dataloaders=_loader(changed_generator),
            ckpt_path=checkpoint.last_model_path,
        )


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    """Verify exact restart outside pytest's RNG-consuming runtime type instrumentation."""
    code = """
import sys
from pathlib import Path
from tests.test_lightning_training import _run_resume_acceptance
_run_resume_acceptance(Path(sys.argv[1]))
"""
    result = subprocess.run(  # noqa: S603 -- fixed current Python executable and script.
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_forward_and_target_transfer_preserve_typed_contract() -> None:
    """Lightning moves frozen target dataclasses and keeps inference target-free."""
    module = _module()
    images, targets = _collate([_NoisyDetectionDataset()[0], _NoisyDetectionDataset()[1]])
    moved_images, moved_targets = module.transfer_batch_to_device(
        (images, targets), torch.device("cpu"), 0
    )
    assert isinstance(moved_targets, list)
    assert all(isinstance(target, DetectionTarget) for target in moved_targets)
    assert all(
        target.labels.device.type == target.boxes.device.type == "cpu" for target in moved_targets
    )
    module.train()
    output = module(moved_images)
    assert output.denoising is None
    assert output.decoder.logits.shape == (2, 2, 2, 3)


def test_optimizer_groups_cover_each_trainable_parameter_once() -> None:
    """Backbone and remaining groups partition all trainable parameters."""
    module = _module(freeze_backbone=False)
    configured: Any = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]
    assert isinstance(scheduler, dict)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(trainable)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    assert {group["lr"] for group in optimizer.param_groups} == {1e-4, 1e-5}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {
        0.0,
        DEFAULT_WEIGHT_DECAY,
    }
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert scheduler["interval"] == "epoch"
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.MultiStepLR)


@pytest.mark.parametrize(
    ("trainer_kwargs", "workers", "match"),
    [
        ({"accumulate_grad_batches": 2}, 0, "accumul"),
        ({}, 1, "num_workers"),
    ],
)
def test_unsupported_trainer_modes_fail_before_training(
    tmp_path: Path, trainer_kwargs: dict[str, Any], workers: int, match: str
) -> None:
    """Unsupported accumulation and worker RNG boundaries fail explicitly."""
    generator = torch.Generator().manual_seed(5)
    trainer = _trainer(tmp_path, epochs=1, generator=generator, **trainer_kwargs)
    with pytest.raises(ValueError, match=match):
        trainer.fit(_module(), train_dataloaders=_loader(generator, workers=workers))


def test_core_and_training_import_without_lightning_installed() -> None:
    """The optional adapter must not make base package imports require Lightning."""
    code = """
import sys
sys.modules['lightning'] = None
sys.modules['lightning.pytorch'] = None
import clean_rtdetr
import clean_rtdetr.training
"""
    result = subprocess.run(  # noqa: S603 -- fixed current Python executable and script.
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
