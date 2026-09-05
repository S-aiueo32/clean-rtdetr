# Copyright 2026 Clean RT-DETR contributors
"""Guard the source tree against reintroducing untyped tensor boundaries."""

import ast
from pathlib import Path

JAXTYPING_ALIASES = {"Bool", "Float", "Float32", "Int64", "Shaped", "UInt8"}
SOURCE_ROOT = Path(__file__).parents[1] / "src" / "clean_rtdetr"


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _annotations(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            found.extend(
                (f"{node.name}() parameter {arg.arg}", arg.annotation)
                for arg in arguments
                if arg.annotation
            )
            if node.args.vararg and node.args.vararg.annotation:
                found.append(
                    (f"{node.name}() parameter {node.args.vararg.arg}", node.args.vararg.annotation)
                )
            if node.args.kwarg and node.args.kwarg.annotation:
                found.append(
                    (f"{node.name}() parameter {node.args.kwarg.arg}", node.args.kwarg.annotation)
                )
            if node.returns:
                found.append((f"{node.name}() return", node.returns))
    return found


def test_tensor_annotations_use_jaxtyping() -> None:
    """Every function annotation mentioning Tensor must use a jaxtyping alias."""
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for location, annotation in _annotations(tree):
            names = _names(annotation)
            if "Tensor" in names and not names & JAXTYPING_ALIASES:
                violations.append(f"{path.relative_to(SOURCE_ROOT)}: {location}")
    assert not violations, "Tensor annotations without jaxtyping: " + ", ".join(violations)
