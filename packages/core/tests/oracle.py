# Copyright 2026 Clean RT-DETR contributors
"""Execute unchanged upstream stage methods without importing its registry/framework.

The checked-in file is a SHA-256-pinned copy. We extract the named method ASTs,
not a rewritten second implementation of the equations under test.
"""

import ast
import copy
import hashlib
import importlib.util
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional, init

from clean_rtdetr.geometry import inverse_sigmoid
from clean_rtdetr.models.attention import deformable_attention_core

UPSTREAM = Path(__file__).resolve().parents[3] / "third_party/RT-DETR/rtdetr_decoder.py"
SHA256 = "c2837740cb66f128b72e802463db57ee3653e15d964baa3b9cf58cc27a817588"
UPSTREAM_UTILS = UPSTREAM.parent / "rtdetr_pytorch/src/zoo/rtdetr/utils.py"
UPSTREAM_UTILS_SHA256 = "a8fe020982e6e5e900a2db3674bc878170d474e12b7c4d1ca17474f95abbd0e3"


class UpstreamStages(nn.Module):
    """Describe the narrow typed surface populated by the unchanged upstream ASTs."""

    eps: float
    eval_spatial_size: tuple[int, int] | None
    num_levels: int
    num_queries: int
    learnt_init_query: bool
    input_proj: nn.ModuleList
    enc_output: nn.Sequential
    enc_score_head: nn.Module
    _generate_anchors: Callable[[Sequence[Sequence[int]]], tuple[Tensor, Tensor]]
    _get_encoder_input: Callable[[tuple[Tensor, ...]], tuple[Tensor, list[list[int]], list[int]]]
    _get_decoder_input: Callable[[Tensor, list[list[int]]], tuple[Tensor, Tensor, Tensor, Tensor]]


def _oracle_deformable_attention_core(
    value: Tensor,
    spatial_shapes: Sequence[Sequence[int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Adapt upstream list metadata to the production tensor-only sampling core."""
    shapes = torch.tensor(spatial_shapes, dtype=torch.int64, device=value.device)
    return deformable_attention_core(value, shapes, sampling_locations, attention_weights)


def stage_oracle() -> UpstreamStages:
    """Load only hash-verified upstream stage methods into a typed module shell."""
    source = UPSTREAM.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256, "Upstream oracle has changed"
    tree = ast.parse(source, filename=str(UPSTREAM))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RTDETRTransformer"
    )
    names = {"_generate_anchors", "_get_decoder_input", "_get_encoder_input"}
    methods: list[ast.stmt] = [
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {"torch": torch, "F": functional}
    # The checked-in source is hash-verified above; no user input reaches exec.
    exec(compile(module, str(UPSTREAM), "exec"), namespace)  # noqa: S102
    oracle_type = type("LoadedStages", (UpstreamStages,), {name: namespace[name] for name in names})
    return cast("UpstreamStages", oracle_type())


def decoder_oracle(  # noqa: PLR0913
    *,
    hidden_dim: int,
    num_heads: int,
    feedforward_dim: int,
    dropout: float,
    activation: str,
    num_levels: int,
    num_points: int,
    num_layers: int,
    eval_index: int,
) -> nn.Module:
    """Load the unchanged upstream decoder classes around the pinned stage loop."""
    source = UPSTREAM.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256, "Upstream oracle has changed"
    tree = ast.parse(source, filename=str(UPSTREAM))
    names = {"MSDeformableAttention", "TransformerDecoderLayer", "TransformerDecoder"}
    classes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert len(classes) == len(names)
    module = ast.Module(body=classes, type_ignores=[])
    namespace: dict[str, object] = {
        "F": functional,
        "copy": copy,
        "deformable_attention_core_func": _oracle_deformable_attention_core,
        "get_activation": {"relu": nn.ReLU, "gelu": nn.GELU}.__getitem__,
        "init": init,
        "inverse_sigmoid": inverse_sigmoid,
        "math": math,
        "nn": nn,
        "OrderedDict": OrderedDict,
        "torch": torch,
    }
    exec(compile(module, str(UPSTREAM), "exec"), namespace)  # noqa: S102
    layer_type = cast("Callable[..., nn.Module]", namespace["TransformerDecoderLayer"])
    decoder_type = cast("Callable[..., nn.Module]", namespace["TransformerDecoder"])
    layer = layer_type(
        hidden_dim,
        num_heads,
        feedforward_dim,
        dropout,
        activation,
        num_levels,
        num_points,
    )
    return decoder_type(hidden_dim, layer, num_layers, eval_index)


def _upstream_utils() -> dict[str, object]:
    """Load the independently pinned upstream utility module for the oracle."""
    if not UPSTREAM_UTILS.is_file():
        msg = f"Missing pinned upstream utility source: {UPSTREAM_UTILS}"
        raise FileNotFoundError(msg)
    if hashlib.sha256(UPSTREAM_UTILS.read_bytes()).hexdigest() != UPSTREAM_UTILS_SHA256:
        msg = "Pinned upstream utility source has changed"
        raise AssertionError(msg)
    spec = importlib.util.spec_from_file_location("clean_rtdetr_upstream_utils", UPSTREAM_UTILS)
    if spec is None or spec.loader is None:
        msg = f"Cannot load {UPSTREAM_UTILS}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


def full_transformer_oracle(**config: object) -> nn.Module:
    """Construct the complete upstream transformer from hash-checked source AST.

    Only the upstream classes are executed.  In particular, this oracle does not
    call clean_rtdetr geometry or attention code, which keeps checkpoint parity
    independent from the implementation under test.
    """
    source = UPSTREAM.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256, "Upstream oracle has changed"
    utility = _upstream_utils()
    tree = ast.parse(source, filename=str(UPSTREAM))
    wanted = {
        "MLP",
        "MSDeformableAttention",
        "TransformerDecoderLayer",
        "TransformerDecoder",
        "RTDETRTransformer",
    }
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
    assert len(classes) == len(wanted), "Pinned decoder is missing an oracle class"
    for node in classes:
        node.decorator_list = []
    module = ast.Module(body=cast("list[ast.stmt]", classes), type_ignores=[])
    namespace: dict[str, object] = {
        "F": functional,
        "copy": copy,
        "init": init,
        "math": math,
        "nn": nn,
        "OrderedDict": OrderedDict,
        "torch": torch,
        "register": lambda value: value,
        **{
            name: utility[name]
            for name in (
                "deformable_attention_core_func",
                "get_activation",
                "inverse_sigmoid",
                "bias_init_with_prob",
            )
        },
    }
    # The source and utility module are repository-pinned; no user input reaches exec.
    exec(compile(module, str(UPSTREAM), "exec"), namespace)  # noqa: S102
    transformer_type = cast("Callable[..., nn.Module]", namespace["RTDETRTransformer"])
    return transformer_type(**config)
