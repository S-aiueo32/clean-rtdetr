# Copyright 2026 Clean RT-DETR contributors
"""Check matching, union localization, and distillation against pinned D-FINE."""

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
from test_distribution import _oracle

from clean_rtdetr.contracts import DetectionOutput, ModelOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, MatchIndices
from clean_rtdetr_training.losses.geometry import box_iou, cxcywh_to_xyxy, generalized_box_iou
from dfine_recipe.criterion import DFINECriterion, union_matches
from dfine_recipe.decoder import DistributionOutput
from dfine_recipe.matcher import DFINEMatcher


def _criterion_oracle() -> Any:  # noqa: ANN401
    """Load upstream loss methods without its registry or optional imports."""
    path = Path(__file__).parent / "reference/dfine_criterion.py.txt"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DFINECriterion"
    )
    cls.decorator_list = []
    names = {
        "loss_labels_vfl",
        "loss_boxes",
        "loss_local",
        "_get_src_permutation_idx",
        "unimodal_distribution_focal_loss",
    }
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    tree.body = [cls]
    namespace: dict[str, Any] = _oracle() | {
        "nn": torch.nn,
        "F": torch.nn.functional,
        "box_cxcywh_to_xyxy": cxcywh_to_xyxy,
        "box_iou": box_iou,
        "generalized_box_iou": generalized_box_iou,
    }
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    criterion = namespace["DFINECriterion"]()
    criterion.alpha, criterion.gamma, criterion.reg_max, criterion.num_classes = 0.75, 2.0, 32, 2
    criterion.fgl_targets = criterion.fgl_targets_dn = None
    criterion.num_pos = criterion.num_neg = None
    return criterion


def _output() -> ModelOutput:
    """Produce fixed geometry with distinct nonuniform layer distributions."""
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn(2, 2, 3, 2, generator=generator, requires_grad=True)
    boxes = torch.full((2, 2, 3, 4), 0.5, requires_grad=True)
    corners = torch.randn(2, 2, 3, 4, 33, generator=generator, requires_grad=True)
    pre = DetectionOutput(logits[0], boxes[0])
    decoded = DistributionOutput(logits, boxes, corners, torch.full((2, 3, 4), 0.5), 4.0, 0.5, pre)
    return ModelOutput(decoded, pre, None)


@pytest.mark.parametrize("empty", [False, True])
def test_losses_and_corner_gradients_match_official_methods(*, empty: bool) -> None:
    """Cover all negatives, mixed batches, FGL interpolation, and detached teacher."""
    output = _output()
    decoded = output.decoder
    assert isinstance(decoded, DistributionOutput)
    no_objects = DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
    targets = [
        no_objects
        if empty
        else DetectionTarget(torch.tensor([1]), torch.tensor([[0.5, 0.5, 0.3, 0.4]])),
        no_objects,
    ]
    criterion = DFINECriterion(DFINEMatcher())
    actual = criterion(output, targets)
    stages = [DetectionOutput(decoded.logits[i], decoded.boxes[i]) for i in range(2)] + [
        decoded.pre,
        output.encoder_aux,
    ]
    assignments = [criterion.matcher(stage, targets) for stage in stages]
    union = union_matches(assignments)
    indices = [(m.prediction_indices, m.target_indices) for m in union]
    union_count = max(sum(len(m.prediction_indices) for m in union), 1)
    oracle = _criterion_oracle()
    upstream_targets = [{"labels": t.labels, "boxes": t.boxes} for t in targets]
    expected = {}
    for i, (stage, suffix) in enumerate(zip(stages, ("_aux_0", "", "_pre", "_enc"), strict=True)):
        values = {"pred_logits": stage.logits, "pred_boxes": stage.boxes}
        own_indices = [(m.prediction_indices, m.target_indices) for m in assignments[i]]
        expected["loss_vfl" + suffix] = oracle.loss_labels_vfl(
            values, upstream_targets, own_indices, 1
        )["loss_vfl"]
        for name, value in oracle.loss_boxes(
            values, upstream_targets, indices, union_count
        ).items():
            expected[name + suffix] = value * (5 if name == "loss_bbox" else 2)
        if i < decoded.logits.shape[0]:
            values.update(
                pred_corners=decoded.corners[i].flatten(-2),
                ref_points=decoded.references,
                teacher_corners=decoded.corners[-1].flatten(-2),
                teacher_logits=decoded.logits[-1],
                reg_scale=torch.tensor([4.0]),
                up=torch.tensor([0.5]),
            )
            for name, value in oracle.loss_local(
                values, upstream_targets, indices, union_count
            ).items():
                expected[name + suffix] = value * (0.15 if name == "loss_fgl" else 1.5)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(sum(actual.values()), decoded.corners, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(sum(expected.values()), decoded.corners)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)


def test_matcher_handles_raw_boxes_and_preserves_normalized_matching() -> None:
    """Relax only the box range contract, keeping Hungarian costs unchanged."""
    targets = [DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.3]]))]
    output = DetectionOutput(
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.3]]]),
    )
    actual = DFINEMatcher()(output, targets)[0]
    expected = HungarianMatcher()(output, targets)[0]
    torch.testing.assert_close(actual.prediction_indices, expected.prediction_indices)
    raw = DetectionOutput(output.logits, output.boxes + 1.0)
    assert DFINEMatcher()(raw, targets)[0].prediction_indices.numel() == 1


def test_distillation_detaches_teacher_and_skips_identical_distributions() -> None:
    """DDF updates the student only; zero-initialized equal layers have zero DDF."""
    output = _output()
    decoded = output.decoder
    assert isinstance(decoded, DistributionOutput)
    targets = [DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))] * 2
    criterion = DFINECriterion(DFINEMatcher())
    loss = criterion(output, targets)["loss_ddf_aux_0"]
    gradient = torch.autograd.grad(loss, decoded.corners)[0]
    assert gradient[0].abs().sum() > 0
    assert gradient[-1].abs().sum() == 0
    with torch.no_grad():
        decoded.corners[0].copy_(decoded.corners[-1])
    assert criterion(output, targets)["loss_ddf_aux_0"] == 0


def test_union_votes_keep_query_unique_and_allow_repeated_targets() -> None:
    """A query's majority target wins; different queries may share that target."""
    stages = [
        [MatchIndices(torch.tensor([0, 1]), torch.tensor([1, 0]))],
        [MatchIndices(torch.tensor([0, 2]), torch.tensor([0, 1]))],
        [MatchIndices(torch.tensor([0, 2]), torch.tensor([1, 1]))],
    ]
    result = union_matches(stages)[0]
    assert dict(
        zip(result.prediction_indices.tolist(), result.target_indices.tolist(), strict=True)
    ) == {0: 1, 1: 0, 2: 1}
