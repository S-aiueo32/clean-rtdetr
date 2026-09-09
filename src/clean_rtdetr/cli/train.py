# Copyright 2026 Clean RT-DETR contributors
"""Run a small Lightning training job and optionally resume its checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from lightning import seed_everything

from clean_rtdetr.configuration import compose_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _compose_config(overrides: list[str]) -> DictConfig:
    """Compose the packaged Lightning configuration through the public API."""
    return compose_config("lightning_training", overrides)


def run(config: DictConfig) -> None:
    """Instantiate and train the detector from the dependency-injection graph."""
    torch.set_num_threads(1)
    seed_everything(int(config.seed), workers=False)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(config.seed))
    datamodule = instantiate(config.datamodule, generator=generator, _convert_="all")
    callbacks = []
    for name, callback_config in config.callbacks.items():
        kwargs = {"generator": generator} if name == "rng" else {}
        callbacks.append(instantiate(callback_config, **kwargs, _convert_="all"))
    trainer = instantiate(config.trainer, callbacks=callbacks, _convert_="all")
    module = instantiate(config.module, _convert_="all")
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.resume, weights_only=False)


def main() -> None:
    """Accept positional Hydra overrides without Hydra's Python 3.14 CLI wrapper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as trainer.max_epochs=1")
    arguments = parser.parse_args()
    run(_compose_config(arguments.overrides))


if __name__ == "__main__":
    main()
