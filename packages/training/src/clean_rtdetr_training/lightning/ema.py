# Copyright 2026 Clean RT-DETR contributors
"""Exponential moving average callback adapted from RT-DETR (Apache-2.0).

Changed: Lightning callback lifecycle and configuration-checked resume. See
third_party/RT-DETR/README.md for pinned upstream provenance.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn


class RTDETREMA(pl.Callback):
    """Keep a frozen, checkpointable EMA copy of ``training_model``.

    The callback assumes the adapter performs one optimizer step per batch.  It
    updates after Lightning's optimizer work in ``on_train_batch_end`` and
    never swaps the EMA weights into the live module.
    """

    def __init__(self, decay: float = 0.9999, warmups: int = 2000) -> None:
        """Configure the EMA decay and warmup ramp."""
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            msg = "decay must be between 0 and 1"
            raise ValueError(msg)
        if type(warmups) is not int or warmups <= 0:
            msg = "warmups must be positive"
            raise ValueError(msg)
        self.decay = decay
        self.warmups = warmups
        self.module: nn.Module | None = None
        self.updates = 0
        self._pending_state: dict[str, Any] | None = None
        self._step_hook_handles: list[Any] = []

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: ANN401
        """Create the EMA copy after the wrapped module is on its device."""
        del trainer
        self.module = copy.deepcopy(pl_module.training_model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(requires_grad=False)
        if self._pending_state is not None:
            self.module.load_state_dict(self._pending_state["module"])
            self.updates = int(self._pending_state["updates"])
            self._pending_state = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: ANN401
        """Update EMA only after an optimizer step actually completes."""
        self._remove_step_hooks()

        def after_step(optimizer: Any, args: Any, kwargs: Any) -> None:  # noqa: ANN401
            del optimizer, args, kwargs
            self._update(pl_module.training_model)

        self._step_hook_handles = [
            optimizer.register_step_post_hook(after_step) for optimizer in trainer.optimizers
        ]

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: ANN401
        """Remove optimizer hooks after training."""
        del trainer, pl_module
        self._remove_step_hooks()

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:  # noqa: ANN401
        """Remove hooks during Trainer teardown."""
        del trainer, pl_module, stage
        self._remove_step_hooks()

    def on_exception(
        self,
        trainer: Any,  # noqa: ANN401
        pl_module: Any,  # noqa: ANN401
        exception: BaseException,
    ) -> None:
        """Release model references when training is interrupted."""
        del trainer, pl_module, exception
        self._remove_step_hooks()

    def _remove_step_hooks(self) -> None:
        for handle in self._step_hook_handles:
            handle.remove()
        self._step_hook_handles = []

    def _update(self, model: nn.Module) -> None:
        """Apply the pinned warmup decay to floating parameters and buffers only."""
        if self.module is None:
            msg = "EMA module is not initialized"
            raise RuntimeError(msg)
        with torch.no_grad():
            self.updates += 1
            factor = self.decay * (1.0 - math.exp(-self.updates / self.warmups))
            source = model.state_dict()
            for name, value in self.module.state_dict().items():
                if value.dtype.is_floating_point:
                    value.mul_(factor)
                    value.add_((1.0 - factor) * source[name].detach())

    def state_dict(self) -> dict[str, Any]:
        """Return EMA weights, update count, and the configured decay parameters."""
        module_state = None if self.module is None else self.module.state_dict()
        return {
            "module": module_state,
            "updates": self.updates,
            "decay": self.decay,
            "warmups": self.warmups,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate configuration and defer weights until ``on_fit_start``."""
        state = state_dict
        if state.get("decay", self.decay) != self.decay:
            msg = "EMA decay does not match the checkpoint"
            raise ValueError(msg)
        if state.get("warmups", self.warmups) != self.warmups:
            msg = "EMA warmups does not match the checkpoint"
            raise ValueError(msg)
        module_state = state.get("module")
        if module_state is None:
            msg = "EMA checkpoint has no module state"
            raise ValueError(msg)
        self._pending_state = {"module": module_state, "updates": int(state.get("updates", 0))}
        self.updates = self._pending_state["updates"]
