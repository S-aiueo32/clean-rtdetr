# Copyright 2026 Clean RT-DETR contributors
"""Short real-device Lightning resume checks."""

# This integration module is intentionally compact and imports CPU fixtures.
# ruff: noqa: PLC0415, PLR0913, PT018, D103, S603

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

pytest.importorskip("lightning")
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecision
from test_lightning_training import (
    _assert_nested_equal,
    _datamodule,
    _module,
    _seed_all,
)

from clean_rtdetr_training.lightning import (
    RTDETREMA,
    EpochBoundaryState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


class _ResumeProbe(Callback):
    """Verify exact restoration before nondeterministic CUDA training resumes."""

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint = checkpoint
        self.checked = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: ANN401
        _assert_nested_equal(
            move_data_to_device(pl_module.state_dict(), "cpu"), self.checkpoint["state_dict"]
        )
        _assert_nested_equal(
            move_data_to_device(trainer.optimizers[0].state_dict(), "cpu"),
            self.checkpoint["optimizer_states"][0],
        )
        for callback in trainer.callbacks:
            if isinstance(callback, RTDETREMA):
                _assert_nested_equal(
                    move_data_to_device(callback.state_dict(), "cpu"),
                    self.checkpoint["callbacks"][callback.state_key],
                )
            if isinstance(callback, EpochBoundaryState):
                saved = self.checkpoint["callbacks"][callback.state_key]
                assert torch.equal(torch.cuda.get_rng_state(pl_module.device), saved["cuda"])
        plugin = trainer.precision_plugin
        if isinstance(plugin, MixedPrecision):
            _assert_nested_equal(plugin.state_dict(), self.checkpoint[type(plugin).__qualname__])
        self.checked = True


def _trainer(
    root: Path,
    generator: torch.Generator,
    precision: str,
    *,
    epochs: int,
    checkpoint: ModelCheckpoint | None = None,
    ema: RTDETREMA | None = None,
) -> Any:  # noqa: ANN401
    import lightning.pytorch as pl

    callbacks: list[Any] = [EpochBoundaryState(generator), ema or RTDETREMA()]
    if checkpoint is not None:
        callbacks.append(checkpoint)
    return pl.Trainer(
        accelerator="cuda",
        devices=1,
        plugins=(
            [MixedPrecision("16-mixed", "cuda", torch.amp.GradScaler("cuda", init_scale=128))]
            if precision == "16-mixed"
            else []
        ),
        deterministic="warn",
        max_epochs=epochs,
        default_root_dir=root,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=checkpoint is not None,
        num_sanity_val_steps=0,
    )


@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
def test_cuda_precision_epoch_resume_preserves_rng_and_updates(
    tmp_path: Path, precision: str, record_property: Callable[[str, object], None]
) -> None:
    code = """
import json
import sys
from pathlib import Path
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, sys.argv[2])
from test_gpu_lightning import _cuda_acceptance
with sdpa_kernel(SDPBackend.MATH):
    print(json.dumps(_cuda_acceptance(Path(sys.argv[1]), sys.argv[3])))
"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(Path(__file__).parent), precision],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record_property("resume_metrics", result.stdout.strip())


def _cuda_acceptance(tmp_path: Path, precision: str) -> dict[str, float | int]:
    """Resume CUDA training and verify optimizer, EMA, and CUDA RNG continuity."""
    total_epochs = 12 if precision == "16-mixed" else 2
    split_epochs = 8 if precision == "16-mixed" else 1
    _seed_all(101)
    initial = {name: value.clone() for name, value in _module().state_dict().items()}

    _seed_all(202)
    full_generator = torch.Generator().manual_seed(303)
    full_module = _module()
    full_module.load_state_dict(initial)
    full_ema = RTDETREMA()
    full_trainer = _trainer(
        tmp_path / "full", full_generator, precision, epochs=total_epochs, ema=full_ema
    )
    full_trainer.fit(full_module, datamodule=_datamodule(full_generator))
    full_cuda_rng = torch.cuda.get_rng_state()

    _seed_all(202)
    split_generator = torch.Generator().manual_seed(303)
    split_module = _module()
    split_module.load_state_dict(initial)
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "split", save_last=True, save_on_train_epoch_end=True
    )
    split_trainer = _trainer(
        tmp_path / "split", split_generator, precision, epochs=split_epochs, checkpoint=checkpoint
    )
    split_trainer.fit(split_module, datamodule=_datamodule(split_generator))
    assert checkpoint.last_model_path

    _seed_all(909)
    resumed_generator = torch.Generator().manual_seed(808)
    resumed_module = _module()
    resumed_ema = RTDETREMA()
    resumed_trainer = _trainer(
        tmp_path / "resumed", resumed_generator, precision, epochs=total_epochs, ema=resumed_ema
    )
    saved_checkpoint = torch.load(
        checkpoint.last_model_path, map_location="cpu", weights_only=False
    )
    assert saved_checkpoint["optimizer_states"][0]["state"]
    probe = _ResumeProbe(saved_checkpoint)
    resumed_trainer.callbacks.append(probe)
    resumed_trainer.fit(
        resumed_module,
        datamodule=_datamodule(resumed_generator),
        ckpt_path=checkpoint.last_model_path,
    )

    assert probe.checked
    # AdamW can amplify CUDA accumulation noise in nearly zero BN bias gradients.
    # Check exact restored state above; bound subsequent numerical drift separately.
    tolerance = 1e-4 if precision == "32-true" else 2e-3
    for name, value in full_module.state_dict().items():
        torch.testing.assert_close(
            value,
            resumed_module.state_dict()[name],
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
    assert full_trainer.global_step == resumed_trainer.global_step == total_epochs
    assert full_ema.module is not None and resumed_ema.module is not None
    assert full_ema.updates == resumed_ema.updates
    assert full_ema.updates > 0, full_trainer.precision_plugin.state_dict()
    for name, value in full_ema.module.state_dict().items():
        torch.testing.assert_close(
            value, resumed_ema.module.state_dict()[name], rtol=tolerance, atol=tolerance
        )
    assert torch.equal(torch.cuda.get_rng_state(), full_cuda_rng)

    assert full_trainer.optimizers[0].state
    assert resumed_trainer.optimizers[0].state
    for full_state, resumed_state in zip(
        full_trainer.optimizers[0].state.values(),
        resumed_trainer.optimizers[0].state.values(),
        strict=True,
    ):
        for key, value in full_state.items():
            torch.testing.assert_close(value, resumed_state[key], rtol=tolerance, atol=tolerance)
    assert (
        full_trainer.lr_scheduler_configs[0].scheduler.state_dict()
        == resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict()
    )
    if precision == "16-mixed":
        assert (
            full_trainer.precision_plugin.scaler.state_dict()
            == resumed_trainer.precision_plugin.scaler.state_dict()
        )

    return {
        "epochs": total_epochs,
        "ema_updates": full_ema.updates,
        "checkpoint_ema_updates": saved_checkpoint["callbacks"][resumed_ema.state_key]["updates"],
        "max_parameter_abs_diff": max(
            float((value - resumed_module.state_dict()[name]).abs().max())
            for name, value in full_module.state_dict().items()
            if value.is_floating_point()
        ),
        "rtol_atol": tolerance,
    }
