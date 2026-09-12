# Copyright 2026 Clean RT-DETR contributors
"""Lightning callback state for reproducible single-process epoch-boundary restart."""

import random
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader


class EpochBoundaryState(Callback):
    """Save application RNG streams alongside Lightning's optimizer and loop states.

    It preserves Python, NumPy, CPU torch, and DataLoader-generator RNG streams,
    plus the active CUDA-device stream when applicable. The supplied CPU generator
    must also be passed to the single training DataLoader. Worker processes and
    mid-epoch checkpoints are outside this contract.
    """

    _CURRENT_VERSION = 2

    def __init__(self, generator: torch.Generator) -> None:
        """Keep the dedicated DataLoader generator separate from model randomness."""
        if generator.device.type != "cpu":
            msg = "The DataLoader generator must use CPU"
            raise ValueError(msg)
        self.generator = generator
        self._pending: dict[str, Any] | None = None
        self._at_boundary = True
        self._batches_seen = 0
        self._cuda_device: torch.device | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore before Lightning constructs the next epoch's DataLoader iterator."""
        del trainer
        self._cuda_device = pl_module.device if pl_module.device.type == "cuda" else None
        if self._pending is not None:
            state = self._pending
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"].cpu())
            numpy_state = state["numpy"]
            np.random.set_state(  # noqa: NPY002 - restore application global RNG state
                (numpy_state[0], numpy_state[1].cpu().numpy(), *numpy_state[2:])
            )
            self.generator.set_state(state["generator"].cpu())
            if state.get("cuda") is not None:
                if pl_module.device.type != "cuda":
                    msg = "CUDA RNG state cannot be restored for a CPU execution"
                    raise ValueError(msg)
                torch.cuda.set_rng_state(state["cuda"], device=pl_module.device)
            elif pl_module.device.type == "cuda":
                msg = "Checkpoint has no CUDA RNG state required for CUDA resume"
                raise ValueError(msg)
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
        cuda_state = None
        # Do not initialize a CUDA context when running the CPU profile.
        if self._cuda_device is not None:
            cuda_state = torch.cuda.get_rng_state(device=self._cuda_device).cpu()
        return {
            "version": self._CURRENT_VERSION,
            "python": random.getstate(),
            "torch": torch.get_rng_state().cpu(),
            "numpy": (
                numpy_state[0],
                torch.from_numpy(numpy_state[1].copy()).cpu(),
                *numpy_state[2:],
            ),
            "generator": self.generator.get_state().cpu(),
            "cuda": cuda_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Delay RNG restoration until model and callback setup have completed."""
        version = state_dict.get("version")
        if version not in {1, 2}:
            msg = "Unsupported epoch RNG checkpoint version"
            raise ValueError(msg)
        if version == self._CURRENT_VERSION and state_dict.get("cuda") is not None:
            state_dict = {**state_dict, "cuda": state_dict["cuda"].cpu()}
        self._pending = state_dict
