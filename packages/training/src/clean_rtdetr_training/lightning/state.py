# Copyright 2026 Clean RT-DETR contributors
"""Lightning callback state for reproducible CPU epoch-boundary restart."""

import random
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader


class EpochBoundaryState(Callback):
    """Save application RNG streams alongside Lightning's optimizer and loop states.

    The supplied CPU generator must also be passed to the single training DataLoader.
    Worker processes and mid-epoch checkpoints are deliberately outside this contract.
    """

    def __init__(self, generator: torch.Generator) -> None:
        """Keep the dedicated DataLoader generator separate from model randomness."""
        if generator.device.type != "cpu":
            msg = "The DataLoader generator must use CPU"
            raise ValueError(msg)
        self.generator = generator
        self._pending: dict[str, Any] | None = None
        self._at_boundary = True
        self._batches_seen = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore before Lightning constructs the next epoch's DataLoader iterator."""
        del trainer, pl_module
        if self._pending is not None:
            state = self._pending
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"])
            numpy_state = state["numpy"]
            np.random.set_state(  # noqa: NPY002 - restore application global RNG state
                (numpy_state[0], numpy_state[1].numpy(), *numpy_state[2:])
            )
            self.generator.set_state(state["generator"])
            self._pending = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Require the DataLoader shape for which RNG restoration has been validated."""
        del pl_module
        loader = trainer.train_dataloader
        if (
            not isinstance(loader, DataLoader)
            or loader.num_workers != 0
            or loader.generator is not self.generator
        ):
            msg = (
                "Epoch resume requires one DataLoader with num_workers=0 and the supplied generator"
            )
            raise ValueError(msg)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Mark active epochs so incomplete iterator state cannot be saved as resumable."""
        del trainer, pl_module
        self._at_boundary = False
        self._batches_seen = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,  # noqa: ANN401 - Lightning hook accepts arbitrary step outputs
        batch: Any,  # noqa: ANN401 - Lightning hook accepts arbitrary batch structures
        batch_idx: int,
    ) -> None:
        """Count completed batches so max_steps cannot masquerade as an epoch boundary."""
        del trainer, pl_module, outputs, batch, batch_idx
        self._batches_seen += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """ModelCheckpoint runs after ordinary callbacks at the completed epoch boundary."""
        del pl_module
        self._at_boundary = self._batches_seen == trainer.num_training_batches

    def state_dict(self) -> dict[str, Any]:
        """Use only tensors and Python primitives in Lightning's callback checkpoint."""
        if not self._at_boundary:
            msg = "Only completed epoch-boundary checkpoints support exact resume"
            raise ValueError(msg)
        numpy_state = np.random.get_state()  # noqa: NPY002 - checkpoint application global RNG
        return {
            "version": 1,
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Delay RNG restoration until model and callback setup have completed."""
        if state_dict.get("version") != 1:
            msg = "Unsupported epoch RNG checkpoint version"
            raise ValueError(msg)
        self._pending = state_dict
