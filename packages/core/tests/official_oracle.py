# Copyright 2026 Clean RT-DETR contributors
"""Independent full RT-DETR oracle used by checkpoint parity checks.

The backbone and encoder are loaded from the pinned upstream source by extracting
their class ASTs.  This keeps registry/framework imports out of the oracle and
ensures parity tests execute the upstream equations directly.
"""

import ast
import collections
import copy
import hashlib
from pathlib import Path
from typing import cast

import torch
from oracle import full_transformer_oracle
from torch import nn

ROOT = Path(__file__).resolve().parents[3] / "third_party/RT-DETR/rtdetr_pytorch/src"
PRESNET_SOURCE = ROOT / "nn/backbone/presnet.py"
BACKBONE_COMMON_SOURCE = ROOT / "nn/backbone/common.py"
HYBRID_ENCODER_SOURCE = ROOT / "zoo/rtdetr/hybrid_encoder.py"
UTILS_SOURCE = ROOT / "zoo/rtdetr/utils.py"
PRESNET_SHA256 = "1faef37155a38d504e2ce3d569e09d8fbbf737c549d1fbb125cee9111fc7bc28"
BACKBONE_COMMON_SHA256 = "29ca34ee34f53ccab66e4a1d985df08ede3df9a686d4f84f861010b8f7513d44"
HYBRID_ENCODER_SHA256 = "3dff40011e125107fd1de2606429f04bc05d52a3d88e742b7ed655c5a07742c5"
UTILS_SHA256 = "a8fe020982e6e5e900a2db3674bc878170d474e12b7c4d1ca17474f95abbd0e3"
UPSTREAM_COMMIT = "068dfde65f2667ad6555883c69d73de886518cad"


def _checked_tree(path: Path, digest: str) -> ast.Module:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        msg = f"Pinned upstream source has changed: {path}"
        raise AssertionError(msg)
    return ast.parse(source, filename=str(path))


def _classes(path: Path, digest: str, names: set[str]) -> list[ast.ClassDef]:
    tree = _checked_tree(path, digest)
    result = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in result} != names:
        msg = f"Pinned source is missing classes: {names}"
        raise AssertionError(msg)
    for node in result:
        node.decorator_list = []
    return result


def _function(path: Path, digest: str, name: str) -> object:
    """Extract one unchanged utility function from a verified source file."""
    tree = _checked_tree(path, digest)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        msg = f"Pinned source is missing function: {name}"
        raise AssertionError(msg)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"nn": nn}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _resnet_cfg() -> object:
    """Load the original ResNet configuration assignment from PResNet source."""
    tree = _checked_tree(PRESNET_SOURCE, PRESNET_SHA256)
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ResNet_cfg"
                for target in node.targets
            )
        ),
        None,
    )
    if assignment is None:
        msg = "Pinned PResNet source is missing ResNet_cfg"
        raise AssertionError(msg)
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        compile(ast.Module(body=[assignment], type_ignores=[]), str(PRESNET_SOURCE), "exec"),
        namespace,
    )
    return namespace["ResNet_cfg"]


def backbone_oracle(**config: object) -> nn.Module:
    """Construct the hash-verified upstream PResNet without registry imports."""
    classes = _classes(
        BACKBONE_COMMON_SOURCE, BACKBONE_COMMON_SHA256, {"ConvNormLayer", "FrozenBatchNorm2d"}
    )
    classes += _classes(
        PRESNET_SOURCE, PRESNET_SHA256, {"BasicBlock", "BottleNeck", "Blocks", "PResNet"}
    )
    module = ast.Module(body=cast("list[ast.stmt]", classes), type_ignores=[])
    namespace = {
        "OrderedDict": collections.OrderedDict,
        "F": torch.nn.functional,
        "ResNet_cfg": _resnet_cfg(),
        "donwload_url": {},
        "get_activation": _function(
            BACKBONE_COMMON_SOURCE, BACKBONE_COMMON_SHA256, "get_activation"
        ),
        "nn": nn,
        "torch": torch,
    }
    exec(compile(module, str(PRESNET_SOURCE), "exec"), namespace)  # noqa: S102
    constructor = cast("type[nn.Module]", namespace["PResNet"])
    return constructor(**config)


def hybrid_encoder_oracle(**config: object) -> nn.Module:
    """Construct the hash-verified upstream HybridEncoder without registry imports."""
    names = {
        "ConvNormLayer",
        "RepVggBlock",
        "CSPRepLayer",
        "TransformerEncoderLayer",
        "TransformerEncoder",
        "HybridEncoder",
    }
    classes = _classes(HYBRID_ENCODER_SOURCE, HYBRID_ENCODER_SHA256, names)
    module = ast.Module(body=cast("list[ast.stmt]", classes), type_ignores=[])
    namespace = {
        "copy": copy,
        "F": torch.nn.functional,
        "get_activation": _function(UTILS_SOURCE, UTILS_SHA256, "get_activation"),
        "nn": nn,
        "torch": torch,
    }
    exec(compile(module, str(HYBRID_ENCODER_SOURCE), "exec"), namespace)  # noqa: S102
    constructor = cast("type[nn.Module]", namespace["HybridEncoder"])
    return constructor(**config)


__all__ = ["UPSTREAM_COMMIT", "backbone_oracle", "full_transformer_oracle", "hybrid_encoder_oracle"]
