# Copyright 2026 Clean RT-DETR contributors
"""Regression tests for local and smoke RT-DETRv4 feature teachers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from rtdetrv4_recipe.teacher import LocalDINOv3Teacher, SmokeTeacher

if TYPE_CHECKING:
    from pathlib import Path


class FakeDINO(nn.Module):
    """Minimal DINOv3-shaped module for local loading tests."""

    embed_dim = 4

    def forward(
        self, images: torch.Tensor, *, is_training: bool, masks: object
    ) -> dict[str, torch.Tensor]:
        """Return spatially constant patch tokens."""
        assert is_training is True
        assert masks is None
        b, _, h, w = images.shape
        return {
            "x_norm_patchtokens": images.mean(1)
            .reshape(b, h // 16, 16, w // 16, 16)
            .mean((2, 4))
            .flatten(1)
            .unsqueeze(-1)
            .expand(b, (h // 16) * (w // 16), 4)
        }


def test_local_teacher_loads_normalizes_and_reshapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify local loading, official normalization, and rectangular reshape."""
    repo = tmp_path / "dinov3"
    repo.mkdir()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture")
    seen: dict[str, object] = {}

    def fake_load(repo_arg: str, name: str, *, source: str, weights: str) -> FakeDINO:
        seen.update(repo=repo_arg, name=name, source=source, weights=weights)
        return FakeDINO()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    teacher = LocalDINOv3Teacher(repo, weights)
    output = teacher(torch.ones(2, 3, 64, 96, requires_grad=True))
    assert output.shape == (2, 4, 2, 3)
    expected = torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]).mean()
    torch.testing.assert_close(output, torch.full_like(output, expected.item()))
    assert output.requires_grad is False
    assert seen == {
        "repo": str(repo),
        "name": "dinov3_vitb16",
        "source": "local",
        "weights": str(weights),
    }
    assert teacher.training is False
    teacher.train(mode=True)
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_local_teacher_validates_files_and_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject missing files, malformed inputs, and malformed teacher tokens."""
    with pytest.raises(FileNotFoundError):
        LocalDINOv3Teacher(tmp_path / "missing", tmp_path / "weights")
    repo = tmp_path / "repo"
    repo.mkdir()
    weights = tmp_path / "weights"
    weights.write_bytes(b"x")
    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: FakeDINO())
    teacher = LocalDINOv3Teacher(repo, weights)
    with pytest.raises(ValueError, match="divisible by 32"):
        teacher(torch.zeros(1, 3, 32, 31))
    with pytest.raises(ValueError, match="floating-point"):
        teacher(torch.zeros(1, 3, 32, 32, dtype=torch.int64))

    class MalformedDINO(FakeDINO):
        """Return an incorrect patch-token count."""

        def forward(
            self, images: torch.Tensor, *, is_training: bool, masks: object
        ) -> dict[str, torch.Tensor]:
            """Return one too few tokens."""
            result = super().forward(images, is_training=is_training, masks=masks)
            return {"x_norm_patchtokens": result["x_norm_patchtokens"][:, :-1]}

    teacher.model = MalformedDINO()
    with pytest.raises(ValueError, match="expected"):
        teacher(torch.zeros(1, 3, 32, 32))


def test_checkpoint_config_contains_weight_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Record an immutable SHA256 checksum in checkpoint provenance."""
    repo = tmp_path / "repo"
    repo.mkdir()
    weights = tmp_path / "weights"
    weights.write_bytes(b"checksum")
    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: FakeDINO())
    teacher = LocalDINOv3Teacher(repo, weights)
    config = teacher.checkpoint_config()
    assert config["weights_sha256"] == hashlib.sha256(b"checksum").hexdigest()
    weights.write_bytes(b"changed")
    assert teacher.checkpoint_config() == config


def test_smoke_teacher_shape_and_checkpoint() -> None:
    """Verify smoke feature geometry and provenance."""
    teacher = SmokeTeacher()
    assert teacher(torch.zeros(1, 3, 64, 96)).shape == (1, 16, 2, 3)
    assert teacher.checkpoint_config() == {"type": "SmokeTeacher", "channels": 16, "stride": 32}
