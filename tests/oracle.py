"""Execute unchanged upstream stage methods without importing its registry/framework.

The checked-in file is a SHA-256-pinned copy. We extract the named method ASTs,
not a rewritten second implementation of the equations under test.
"""

import ast
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party/RT-DETR/rtdetr_decoder.py"
SHA256 = "c2837740cb66f128b72e802463db57ee3653e15d964baa3b9cf58cc27a817588"


def stage_oracle() -> nn.Module:
    source = UPSTREAM.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256, "Upstream oracle has changed"
    tree = ast.parse(source, filename=str(UPSTREAM))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RTDETRTransformer"
    )
    names = {"_generate_anchors", "_get_decoder_input", "_get_encoder_input"}
    methods = [
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {"torch": torch, "F": F}
    exec(compile(module, str(UPSTREAM), "exec"), namespace)
    oracle_type = type("UpstreamStages", (nn.Module,), {name: namespace[name] for name in names})
    return oracle_type()
