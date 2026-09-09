# Copyright 2026 Clean RT-DETR contributors
"""Keep supervision state and framework dependencies outside the model package."""

import ast
from dataclasses import fields
from pathlib import Path

import clean_rtdetr
from clean_rtdetr import contracts


def test_core_contracts_do_not_own_supervision_assignments() -> None:
    """Only the training package may expose DN assignments and grouped predictions."""
    training_names = {"DenoisingBatch", "DenoisingMetadata", "DenoisingOutput", "TrainingOutput"}
    assert training_names.isdisjoint(vars(contracts))
    assert training_names.isdisjoint(vars(clean_rtdetr))
    assert "denoising" not in {field.name for field in fields(contracts.ModelOutput)}


def test_core_imports_do_not_depend_on_training_frameworks() -> None:
    """Enforce the package dependency direction even for deferred imports."""
    forbidden = {"clean_rtdetr_training", "lightning", "hydra", "omegaconf", "scipy"}
    source = Path(__file__).parents[1] / "src" / "clean_rtdetr"
    violations = []
    for path in source.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(module.split(".")[0] in forbidden for module in modules):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, f"Core imports training dependencies: {violations}"
