# Copyright 2026 Clean RT-DETR contributors
"""Hash-checked, dependency-light references for inference boundaries."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Callable


ROOT = Path(__file__).resolve().parents[1] / "third_party/RT-DETR/rtdetr_pytorch"
POSTPROCESSOR_SOURCE = ROOT / "src/zoo/rtdetr/rtdetr_postprocessor.py"
BOX_OPS_SOURCE = ROOT / "src/zoo/rtdetr/box_ops.py"
TRANSFORMS_SOURCE = ROOT / "src/data/transforms.py"
DATALOADER_SOURCE = ROOT / "configs/rtdetr/include/dataloader.yml"
UPSTREAM_COMMIT = "068dfde65f2667ad6555883c69d73de886518cad"
PINNED_SOURCES = {
    POSTPROCESSOR_SOURCE: "a33a3afbc955866eb08587fdc77da0ae446153af952f42b721abc981f413444c",
    BOX_OPS_SOURCE: "7f39e132468a32de058786b01256bc2d801b4f28600fba6b0496ed22678a1a1f",
    TRANSFORMS_SOURCE: "eccf323d38ee2ab2eddfe18ea064a9fd30801fa926123f9d85fe2529803577c3",
    DATALOADER_SOURCE: "be594d0ee141b74e47368afff5674b2212b8c3e849163a6b821fb901e58f6d3c",
}


def assert_pinned_sources() -> None:
    """Fail loudly if any checked-in upstream provenance file was edited."""
    for path, expected in PINNED_SOURCES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Pinned upstream source has changed: {path}"


def _box_cxcywh_to_xyxy(
    boxes: Tensor, *, in_fmt: str, out_fmt: str, converter: Callable[[Tensor], Tensor]
) -> Tensor:
    """Adapt the independently extracted upstream converter to torchvision's API."""
    assert (in_fmt, out_fmt) == ("cxcywh", "xyxy")
    return converter(boxes)


def postprocessor_oracle(*, num_classes: int, num_top_queries: int) -> SimpleNamespace:
    """Extract the unchanged upstream ``forward`` into a small shell object."""
    assert_pinned_sources()
    tree = ast.parse(POSTPROCESSOR_SOURCE.read_bytes(), filename=str(POSTPROCESSOR_SOURCE))
    processor = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    forward = next(
        node
        for node in processor.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    module = ast.Module(body=[forward], type_ignores=[])
    ast.fix_missing_locations(module)
    box_tree = ast.parse(BOX_OPS_SOURCE.read_bytes(), filename=str(BOX_OPS_SOURCE))
    box_function = next(
        node
        for node in box_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "box_cxcywh_to_xyxy"
    )
    box_module = ast.Module(body=[box_function], type_ignores=[])
    ast.fix_missing_locations(box_module)
    box_namespace = {"torch": torch}
    exec(compile(box_module, str(BOX_OPS_SOURCE), "exec"), box_namespace)  # noqa: S102
    converter = cast("Callable[[Tensor], Tensor]", box_namespace["box_cxcywh_to_xyxy"])
    namespace = {
        "F": functional,
        "torch": torch,
        "torchvision": SimpleNamespace(
            ops=SimpleNamespace(
                box_convert=lambda boxes, *, in_fmt, out_fmt: _box_cxcywh_to_xyxy(
                    boxes, in_fmt=in_fmt, out_fmt=out_fmt, converter=converter
                ),
            ),
        ),
    }
    exec(compile(module, str(POSTPROCESSOR_SOURCE), "exec"), namespace)  # noqa: S102
    shell = SimpleNamespace(
        use_focal_loss=True,
        num_top_queries=num_top_queries,
        num_classes=num_classes,
        deploy_mode=False,
        remap_mscoco_category=False,
    )
    shell.forward = cast("FunctionType", namespace["forward"]).__get__(shell)
    return shell


def preprocess_reference(image: Image.Image) -> Tensor:
    """Apply the documented official validation path to one PIL image.

    This deliberately models only RGB conversion, 640px bilinear resize, raw
    byte decoding, and division by 255; it does not import the old transform
    registry or claim to execute the complete upstream data pipeline.
    """
    assert_pinned_sources()
    if not isinstance(image, Image.Image):
        msg = "reference preprocessing expects a PIL image"
        raise TypeError(msg)
    rgb = image.convert("RGB").resize((640, 640), resample=Image.Resampling.BILINEAR)
    return (
        torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        .reshape(rgb.height, rgb.width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        .div(255)
    )


__all__ = ["PINNED_SOURCES", "UPSTREAM_COMMIT", "postprocessor_oracle", "preprocess_reference"]
