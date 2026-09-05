# Copyright 2026 Clean RT-DETR contributors
"""Hash-checked references for the training boundary.

The reference objects below are deliberately extracted from the pinned upstream
criterion.  The production implementation has a small, typed API; this module
only adapts that API to the dictionaries expected by the upstream code.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1] / "third_party/RT-DETR/rtdetr_pytorch"
MATCHER_SOURCE = ROOT / "src/zoo/rtdetr/matcher.py"
CRITERION_SOURCE = ROOT / "src/zoo/rtdetr/rtdetr_criterion.py"
BOX_OPS_SOURCE = ROOT / "src/zoo/rtdetr/box_ops.py"
PINNED_SOURCES = {
    MATCHER_SOURCE: "7e364fbba500b0ab23e3e33b7430780cd8dcf4738bd3b961118a3b1f8acd020e",
    CRITERION_SOURCE: "54e0f25364bf5c52ae60f8f9363dd2df7d9274657d97a881a2757b7805b739df",
    BOX_OPS_SOURCE: "7f39e132468a32de058786b01256bc2d801b4f28600fba6b0496ed22678a1a1f",
}


def assert_pinned_sources() -> None:
    """Ensure the provenance files used by this oracle remain unchanged."""
    for path, expected in PINNED_SOURCES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
            f"Pinned upstream source has changed: {path}"
        )


def _extract_class(path: Path, name: str, namespace: dict[str, Any]) -> type[Any]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    # The upstream registration decorator imports the complete application
    # registry.  It is intentionally removed from this dependency-light oracle.
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return cast("type[Any]", namespace[name])


def _namespace() -> dict[str, Any]:
    box_tree = ast.parse(BOX_OPS_SOURCE.read_bytes(), filename=str(BOX_OPS_SOURCE))
    functions: list[ast.stmt] = [
        node
        for node in box_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"box_cxcywh_to_xyxy", "box_iou", "generalized_box_iou"}
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    # torchvision's box_area is a three-line arithmetic helper; keeping it
    # here avoids importing a compiled torchvision operator in CPU tests.
    def box_area(boxes: Tensor) -> Tensor:
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "box_area": box_area,
    }
    exec(compile(module, str(BOX_OPS_SOURCE), "exec"), namespace)  # noqa: S102
    namespace.update(
        {
            "linear_sum_assignment": linear_sum_assignment,
            "get_world_size": lambda: 1,
            "is_dist_available_and_initialized": lambda: False,
        }
    )
    return namespace


def _objects() -> tuple[type[Any], type[Any]]:
    assert_pinned_sources()
    matcher_namespace = _namespace()
    matcher = _extract_class(MATCHER_SOURCE, "HungarianMatcher", matcher_namespace)
    criterion_namespace = _namespace()
    criterion = _extract_class(CRITERION_SOURCE, "SetCriterion", criterion_namespace)
    return matcher, criterion


def criterion_oracle(  # noqa: PLR0913
    output: object,
    targets: object,
    *,
    weight_vfl: float = 1.0,
    weight_bbox: float = 5.0,
    weight_giou: float = 2.0,
    alpha: float = 0.75,
    gamma: float = 2.0,
    matcher_alpha: float = 0.25,
    matcher_gamma: float = 2.0,
) -> dict[str, Tensor]:
    """Evaluate pinned upstream VFL/L1/GIoU for a DetectionOutput or ModelOutput."""
    matcher_type, criterion_type = _objects()
    output = cast("Any", output)
    targets = cast("Any", targets)
    matcher = matcher_type(
        {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=matcher_alpha,
        gamma=matcher_gamma,
    )
    criterion = criterion_type(
        matcher,
        {"loss_vfl": weight_vfl, "loss_bbox": weight_bbox, "loss_giou": weight_giou},
        ["vfl", "boxes"],
        alpha=alpha,
        gamma=gamma,
        num_classes=(
            output.final.logits.shape[-1] if hasattr(output, "final") else output.logits.shape[-1]
        ),
    )

    final = output.final if hasattr(output, "final") else output
    converted: dict[str, Any] = {"pred_logits": final.logits, "pred_boxes": final.boxes}
    if hasattr(output, "decoder"):
        converted["aux_outputs"] = [
            {"pred_logits": logits, "pred_boxes": boxes}
            for logits, boxes in zip(
                output.decoder.logits[:-1], output.decoder.boxes[:-1], strict=True
            )
        ]
        converted["aux_outputs"].append(
            {"pred_logits": output.encoder_aux.logits, "pred_boxes": output.encoder_aux.boxes}
        )
        if output.denoising is not None:
            converted["dn_aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(
                    output.denoising.decoder.logits,
                    output.denoising.decoder.boxes,
                    strict=True,
                )
            ]
            converted["dn_meta"] = {
                "dn_positive_idx": output.denoising.metadata.positive_indices,
                "dn_num_group": output.denoising.metadata.num_groups,
            }
    upstream_targets = [{"labels": target.labels, "boxes": target.boxes} for target in targets]
    return criterion(converted, upstream_targets)


def matcher_oracle(output: object, targets: object) -> tuple[tuple[Tensor, Tensor], ...]:
    """Return upstream focal Hungarian indices for a typed prediction boundary."""
    matcher_type, _ = _objects()
    output = cast("Any", output)
    targets = cast("Any", targets)
    matcher = matcher_type(
        {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    upstream_output = {"pred_logits": output.logits, "pred_boxes": output.boxes}
    upstream_targets = [{"labels": t.labels, "boxes": t.boxes} for t in targets]
    return tuple(matcher(upstream_output, upstream_targets))


__all__ = ["PINNED_SOURCES", "assert_pinned_sources", "criterion_oracle", "matcher_oracle"]
