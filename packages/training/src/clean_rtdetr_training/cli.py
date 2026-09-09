# Copyright 2026 Clean RT-DETR contributors
"""Run a small Lightning training job and optionally resume its checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import torch
import typer
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf

from clean_rtdetr_training.configuration import compose_training_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(config: DictConfig) -> None:
    """Instantiate and train the detector from the dependency-injection graph."""
    torch.set_num_threads(1)
    seed_everything(int(config.seed), workers=False)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(config.seed))
    module = instantiate(config.module, _convert_="all")
    datamodule = instantiate(
        config.datamodule,
        training_model=module.training_model,
        generator=generator,
        _convert_="all",
    )
    callbacks = []
    for name, callback_config in config.trainer.get("callbacks", {}).items():
        kwargs = {"generator": generator} if name == "rng" else {}
        callbacks.append(instantiate(callback_config, **kwargs, _convert_="all"))
    # Callback recipes are a named mapping; Trainer expects the constructed list.
    trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
    if not isinstance(trainer_config, dict):
        msg = "Trainer configuration must be a mapping"
        raise TypeError(msg)
    trainer_config.pop("callbacks", None)
    trainer = instantiate(trainer_config, callbacks=callbacks, _convert_="all")
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.resume, weights_only=False)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def train(
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing the Hydra configuration tree.",
        ),
    ],
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra configuration name without .yaml.")
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides such as trainer.max_epochs=1"),
    ] = None,
) -> None:
    """Accept positional Hydra overrides without Hydra's Python 3.14 CLI wrapper."""
    run(compose_training_config(config_name, overrides or [], config_dir=config_dir))


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
