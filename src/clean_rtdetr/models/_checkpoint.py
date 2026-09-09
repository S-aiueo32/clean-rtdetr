# Copyright 2026 Clean RT-DETR contributors
"""Describe injected module behavior without guessing arbitrary Python state."""

from torch import nn


def module_checkpoint_config(module: nn.Module) -> dict[str, object]:
    """Require each concrete extension type to describe its non-tensor settings."""
    module_type = type(module)
    target = f"{module_type.__module__}.{module_type.__qualname__}"
    if module_type is nn.Linear:
        config = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    else:
        describe = module_type.__dict__.get("checkpoint_config")
        if not callable(describe):
            msg = f"Injected module {target} must explicitly implement checkpoint_config()"
            raise TypeError(msg)
        config = describe(module)
        if not isinstance(config, dict):
            msg = f"Injected module {target}.checkpoint_config() must return a dict"
            raise TypeError(msg)
    return {"target": target, "config": config}
