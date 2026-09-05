# Copyright 2026 Clean RT-DETR contributors
"""Run a small Lightning training job and optionally resume its checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import seed_everything

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _compose_config(overrides: list[str]) -> DictConfig:
    """Compose the example config through Hydra's Python API."""
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="lightning_training", overrides=overrides)


def run(config: DictConfig) -> None:
    """Instantiate and train the detector from the dependency-injection graph."""
    torch.set_num_threads(1)
    seed_everything(int(config.seed), workers=False)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(config.seed))
    dataset = instantiate(config.dataset, _convert_="all")
    collate_fn = instantiate(config.collate, _convert_="all")
    loader = instantiate(
        config.dataloader,
        dataset=dataset,
        generator=generator,
        collate_fn=collate_fn,
        _convert_="all",
    )
    callbacks = []
    for name, callback_config in config.callbacks.items():
        kwargs = {"generator": generator} if name == "rng" else {}
        callbacks.append(instantiate(callback_config, **kwargs, _convert_="all"))
    trainer = instantiate(config.trainer, callbacks=callbacks, _convert_="all")
    module = instantiate(config.module, _convert_="all")
    trainer.fit(module, train_dataloaders=loader, ckpt_path=config.resume, weights_only=False)


def main() -> None:
    """Accept positional Hydra overrides without Hydra's Python 3.14 CLI wrapper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as trainer.max_epochs=1")
    arguments = parser.parse_args()
    run(_compose_config(arguments.overrides))


if __name__ == "__main__":
    main()
