# Copyright 2026 Clean RT-DETR contributors
"""Tests for the optional Lightning EMA callback."""

import ast
import hashlib
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning.pytorch as pl
import pytest
import torch
from torch import nn

from clean_rtdetr_training.lightning import RTDETREMA


class TinyTrainingModule(pl.LightningModule):
    """Minimal wrapper exposing the adapter's training model contract."""

    def __init__(self) -> None:
        """Create a model with floating and integer state."""
        super().__init__()
        self.training_model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.training_model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))


def _update(callback: RTDETREMA, wrapper: TinyTrainingModule) -> None:
    callback.on_train_batch_end(None, wrapper, None, None, 0)


def _oracle(model: nn.Module) -> Any:  # noqa: ANN401 - unmodified upstream dynamic class
    source = (
        Path(__file__).resolve().parents[3] / "third_party/RT-DETR/rtdetr_pytorch/src/optim/ema.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "9ea3e2d870adb531d19da8f33d2603ee884a65f3ff57583aaaac7321e23f0083"
    )
    node = next(
        node
        for node in ast.parse(source.read_bytes()).body
        if isinstance(node, ast.ClassDef) and node.name == "ModelEMA"
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "math": math,
        "deepcopy": deepcopy,
        "dist": SimpleNamespace(de_parallel=lambda model: model),
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - pinned test oracle
    return namespace["ModelEMA"](model, decay=0.8, warmups=2)


def test_ema_matches_upstream_formula_and_freezes_integer_buffers() -> None:
    """Match the warmup formula while leaving integer buffers unchanged."""
    torch.manual_seed(4)
    wrapper = TinyTrainingModule()
    callback = RTDETREMA(decay=0.8, warmups=2)
    callback.on_fit_start(None, wrapper)
    assert callback.module is not None
    oracle = _oracle(wrapper.training_model)
    initial = {name: value.clone() for name, value in wrapper.training_model.state_dict().items()}
    expected_state = {
        name: value.clone() for name, value in initial.items() if value.dtype.is_floating_point
    }

    for step in range(1, 3):
        with torch.no_grad():
            for parameter in wrapper.training_model.parameters():
                parameter.add_(1.0)
            wrapper.training_model.get_buffer("counter").add_(1)
        _update(callback, wrapper)
        oracle.update(wrapper.training_model)
        factor = 0.8 * (1.0 - math.exp(-step / 2))
        for name, value in callback.module.state_dict().items():
            torch.testing.assert_close(value, oracle.module.state_dict()[name], rtol=0, atol=0)
            if value.dtype.is_floating_point:
                expected_state[name] = (
                    factor * expected_state[name]
                    + (1.0 - factor) * wrapper.training_model.state_dict()[name]
                )
                torch.testing.assert_close(value, expected_state[name])
            else:
                torch.testing.assert_close(value, initial[name])
    assert callback.updates == 2  # noqa: PLR2004
    assert all(not parameter.requires_grad for parameter in callback.module.parameters())


def test_ema_state_can_resume_and_validates_configuration() -> None:
    """Restore EMA weights and reject incompatible callback configuration."""
    torch.manual_seed(7)
    wrapper = TinyTrainingModule()
    callback = RTDETREMA(decay=0.9, warmups=3)
    callback.on_fit_start(None, wrapper)
    _update(callback, wrapper)
    state = callback.state_dict()

    resumed = RTDETREMA(decay=0.9, warmups=3)
    resumed.load_state_dict(state)
    resumed.on_fit_start(None, wrapper)
    assert resumed.module is not None
    assert callback.module is not None
    assert resumed.updates == 1
    for name, value in resumed.module.state_dict().items():
        torch.testing.assert_close(value, callback.module.state_dict()[name])

    with pytest.raises(ValueError, match="decay"):
        RTDETREMA(decay=0.8, warmups=3).load_state_dict(state)
    with pytest.raises(ValueError, match="warmups"):
        RTDETREMA(decay=0.9, warmups=2).load_state_dict(state)
