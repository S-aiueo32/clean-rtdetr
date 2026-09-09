# Copyright 2026 Clean RT-DETR contributors
"""Hash-checked adapter around the pinned upstream DN query builder."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from clean_rtdetr.core.geometry import inverse_sigmoid

if TYPE_CHECKING:
    from clean_rtdetr_training.contracts import DetectionTarget

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "third_party/RT-DETR/rtdetr_pytorch/src/zoo/rtdetr/denoising.py"
)
SOURCE_SHA256 = "2254d6b781879f54684e5726ada5e9c99414873398cf0a42c2e11d691bece595"


def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    x_c, y_c, width, height = boxes.unbind(-1)
    return torch.stack(
        (x_c - 0.5 * width, y_c - 0.5 * height, x_c + 0.5 * width, y_c + 0.5 * height),
        dim=-1,
    )


def _box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def _upstream_function() -> Any:  # noqa: ANN401
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    tree = ast.parse(SOURCE.read_bytes(), filename=str(SOURCE))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "get_contrastive_denoising_training_group"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "torch": torch,
        "inverse_sigmoid": inverse_sigmoid,
        "box_cxcywh_to_xyxy": _box_cxcywh_to_xyxy,
        "box_xyxy_to_cxcywh": _box_xyxy_to_cxcywh,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["get_contrastive_denoising_training_group"]


def denoising_oracle(  # noqa: PLR0913
    targets: list[DetectionTarget],
    *,
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int,
    label_noise_ratio: float,
    box_noise_scale: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]] | tuple[None, None, None, None]:
    """Run the unchanged upstream function against typed targets."""
    converted = [{"labels": target.labels, "boxes": target.boxes} for target in targets]
    function = _upstream_function()
    return cast(
        "tuple[Tensor, Tensor, Tensor, dict[str, Any]] | tuple[None, None, None, None]",
        function(
            converted,
            num_classes,
            num_queries,
            class_embed,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        ),
    )
