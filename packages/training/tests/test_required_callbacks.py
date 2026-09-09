# Copyright 2026 Clean RT-DETR contributors
"""User callback requirements survive checkpoints and configuration changes."""

from functools import partial
from typing import Any, cast

import pytest
import torch
from jaxtyping import TypeCheckError
from lightning.pytorch import Callback, Trainer
from test_training_interfaces import ToyCriterion, ToyTrainingModel, _loader

from clean_rtdetr_training.lightning import RTDETRLightningModule


class AdaptiveCallback(Callback):
    """Represent recipe state whose presence changes training behavior."""

    @property
    def state_key(self) -> str:
        """Identify an arbitrary caller-owned callback."""
        return "user.AdaptiveCallback"


def _module(keys: tuple[str, ...] = ()) -> RTDETRLightningModule:
    return RTDETRLightningModule(
        ToyTrainingModel(),
        ToyCriterion(),
        partial(torch.optim.SGD, lr=0.1),
        required_callback_keys=keys,
    )


def _trainer(*callbacks: Callback) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        callbacks=list(callbacks),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def test_required_callback_keys_survive_removed_declaration() -> None:
    """A saved requirement remains mandatory across multiple resume generations."""
    key = AdaptiveCallback().state_key
    original = _module((key,))
    original.trainer = _trainer(AdaptiveCallback())
    checkpoint: dict[str, Any] = {}
    original.on_save_checkpoint(checkpoint)

    restored = _module()
    restored.on_load_checkpoint(checkpoint)
    restored.trainer = _trainer()
    with pytest.raises(ValueError, match=r"Resume requires.*user.AdaptiveCallback"):
        restored.on_fit_start()

    restored.trainer = _trainer(AdaptiveCallback())
    restored.on_fit_start()
    next_checkpoint: dict[str, Any] = {}
    restored.on_save_checkpoint(next_checkpoint)
    assert next_checkpoint["clean_rtdetr_required_callbacks"] == [key]


def test_callback_requirement_is_checked_before_training() -> None:
    """A fresh run fails before updating parameters when its callback is missing."""
    module = _module((AdaptiveCallback().state_key,))
    initial = {name: value.clone() for name, value in module.state_dict().items()}
    with pytest.raises(ValueError, match=r"Training requires.*user.AdaptiveCallback"):
        _trainer().fit(module, train_dataloaders=_loader())
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_existing_checkpoints_still_require_builtin_callbacks() -> None:
    """Older metadata retains its EMA/RNG checks without requiring logging callbacks."""
    original = _module()
    original.trainer = _trainer()
    checkpoint: dict[str, Any] = {}
    original.on_save_checkpoint(checkpoint)
    del checkpoint["clean_rtdetr_required_callbacks"]
    checkpoint["callbacks"] = {"EpochBoundaryState": {}, "ModelCheckpoint": {}}
    restored = _module()
    restored.on_load_checkpoint(checkpoint)
    restored.trainer = _trainer()
    with pytest.raises(ValueError, match=r"Resume requires.*EpochBoundaryState"):
        restored.on_fit_start()


@pytest.mark.parametrize("keys", ["callback", ("",), (None,)])
def test_callback_keys_reject_malformed_configuration(keys: object) -> None:
    """Reject a bare string or invalid key instead of treating it as characters."""
    with pytest.raises((ValueError, TypeCheckError), match=r"callback.keys"):
        _module(cast("tuple[str, ...]", keys))
