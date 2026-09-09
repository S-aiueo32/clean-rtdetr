# Copyright 2026 Clean RT-DETR contributors
"""Check dataset injection and exact epoch-boundary resume through the real CLI."""

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from clean_rtdetr_training import compose_training_config
from clean_rtdetr_training.cli import run
from clean_rtdetr_training.data import PennFudanDetectionDataset, SyntheticDetectionDataset


def _config(overrides: list[str]) -> DictConfig:
    """Resolve the shipped recipe, selecting the explicit random smoke teacher."""
    return compose_training_config(
        "train", ["teacher=smoke", *overrides], config_dir=Path(__file__).parents[1] / "configs"
    )


def test_penn_fudan_and_synthetic_injection(tmp_path: Path) -> None:
    """Use both installed datasets without copying their loading or metadata logic."""
    (tmp_path / "PNGImages").mkdir()
    (tmp_path / "PedMasks").mkdir()
    for index in range(3):
        Image.new("RGB", (10, 6), "red").save(tmp_path / "PNGImages" / f"person{index}.png")
        mask = Image.new("L", (10, 6))
        ImageDraw.Draw(mask).rectangle((2, 1, 5, 3), fill=1)
        mask.save(tmp_path / "PedMasks" / f"person{index}_mask.png")
    config = _config([f"data.root={tmp_path}", "data.val_size=1"])
    dataset = instantiate(config.data)()
    assert isinstance(dataset, PennFudanDetectionDataset)
    assert len(dataset) == 2  # noqa: PLR2004
    image, target = dataset[0]
    assert image.shape == (3, 640, 640)
    torch.testing.assert_close(target.labels, torch.tensor([0]))
    synthetic = instantiate(_config(["data=synthetic"]).data)()
    assert isinstance(synthetic, SyntheticDetectionDataset)
    assert synthetic.num_classes == 1


def _run(output: Path, *, epochs: int, resume: Path | None = None, **kwargs: object) -> Path:
    """Train a compact student using all shipped loss/callback/CLI boundaries."""
    config = _config(["data=synthetic", f"output_dir={output}", f"trainer.max_epochs={epochs}"])
    config.data.num_classes = 1
    config.module.training_model = {"_target_": "test_model.make_model"}
    for key, value in kwargs.items():
        config.trainer.callbacks.gam[key] = value
    if resume is not None:
        config.resume = str(resume)
    run(config)
    return output / "checkpoints" / "last.ckpt"


def test_epoch_boundary_resume_preserves_student_projector_and_gam(tmp_path: Path) -> None:
    """A resumed run reproduces uninterrupted parameters and adaptive weights exactly."""
    full_path = _run(tmp_path / "full", epochs=2)
    first_path = _run(tmp_path / "split", epochs=1)
    first = torch.load(first_path, weights_only=False)
    resumed_path = _run(tmp_path / "split", epochs=2, resume=first_path)
    full = torch.load(full_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    assert "GradientAdaptiveModulation" in resumed["callbacks"]
    assert first["state_dict"]["criterion.distillation_weight"].item() != 15.0  # noqa: PLR2004
    for key, value in full["state_dict"].items():
        torch.testing.assert_close(value, resumed["state_dict"][key], rtol=0, atol=0)
    assert (
        full["callbacks"]["GradientAdaptiveModulation"]
        == (resumed["callbacks"]["GradientAdaptiveModulation"])
    )
    with pytest.raises(ValueError, match="GAM setting rho"):
        _run(tmp_path / "changed", epochs=3, resume=resumed_path, rho=3.0)
    config = _config(
        ["data=synthetic", f"output_dir={tmp_path / 'missing'}", "trainer.max_epochs=3"]
    )
    config.data.num_classes = 1
    config.module.training_model = {"_target_": "test_model.make_model"}
    config.resume = str(resumed_path)
    del config.trainer.callbacks.gam
    with pytest.raises(ValueError, match=r"Resume requires.*GradientAdaptiveModulation"):
        run(config)
