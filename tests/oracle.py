# Copyright 2026 Clean RT-DETR contributors
"""Execute unchanged upstream stage methods without importing its registry/framework.

The checked-in file is a SHA-256-pinned copy. We extract the named method ASTs,
not a rewritten second implementation of the equations under test.
"""

import ast
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party/RT-DETR/rtdetr_decoder.py"
SHA256 = "c2837740cb66f128b72e802463db57ee3653e15d964baa3b9cf58cc27a817588"


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
