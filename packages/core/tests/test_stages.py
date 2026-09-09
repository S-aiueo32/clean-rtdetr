# Copyright 2026 Clean RT-DETR contributors
"""Verify tensor contracts, constructor injection, and parity with pinned upstream stages."""

import copy
from dataclasses import replace

import pytest
import torch
from jaxtyping import TypeCheckError
from oracle import stage_oracle
from torch import nn

from clean_rtdetr import (
    FeatureSpec,
    MemoryProjector,
    MultiScaleFeatures,
    QuerySelector,
    QuerySelectorConfig,
)
from clean_rtdetr.core.contracts import EncodedFeatures
from clean_rtdetr.core.geometry import generate_anchors, inverse_sigmoid
from clean_rtdetr.models.selector import BoxHead


def encoded(
    shapes: tuple[tuple[int, int], ...] = ((7, 9), (3, 5), (2, 2)),
    channels: int = 8,
    batch: int = 2,
) -> EncodedFeatures:
    """Create non-square multi-level memory with differentiable random features."""
    metadata = torch.tensor(shapes, dtype=torch.int64)
    sizes = metadata.prod(1)
    memory = torch.randn(batch, int(sizes.sum()), channels, requires_grad=True)
    return EncodedFeatures(memory, metadata, sizes.cumsum(0) - sizes)


@pytest.mark.parametrize("shapes", [((80, 80), (40, 40), (20, 20)), ((80, 3), (5, 7))])
def test_anchors_match_pinned_upstream_including_invalid_positions(
    shapes: tuple[tuple[int, int], ...],
) -> None:
    """Match anchor logits, validity masks, and intended infinities exactly."""
    oracle = stage_oracle()
    oracle.eps = 0.01
    actual, valid = generate_anchors(shapes)
    expected, expected_valid = oracle._generate_anchors(shapes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(valid, expected_valid)
    assert (~valid).any()
    assert torch.isposinf(actual[~valid.expand_as(actual)]).all()
    assert not torch.isnan(actual).any()


def test_flatten_order_offsets_and_projection_match_upstream() -> None:
    """Preserve projection values and W-first flatten order across feature levels."""
    torch.manual_seed(7)
    specs = (FeatureSpec(3, 8), FeatureSpec(5, 16))
    levels = (torch.randn(2, 3, 5, 7), torch.randn(2, 5, 3, 4))
    projector = MemoryProjector(specs, hidden_dim=8).eval()
    oracle = stage_oracle()
    oracle.input_proj = copy.deepcopy(projector.input_proj)
    oracle.num_levels = 2
    oracle.eval()
    actual = projector(MultiScaleFeatures(levels, specs))
    expected, shapes, offsets = oracle._get_encoder_input(levels)
    actual.validate()
    torch.testing.assert_close(actual.memory, expected, rtol=0, atol=0)
    assert actual.spatial_shapes.tolist() == shapes == [[5, 7], [3, 4]]
    assert actual.level_start_index.tolist() == offsets == [0, 35]
    projected_first = projector.input_proj[0](levels[0])
    torch.testing.assert_close(actual.memory[:, 1], projected_first[:, :, 0, 1])


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_selection_outputs_and_gradients_match_pinned_upstream(
    *,
    learned: bool,
    training: bool,
) -> None:
    """Match upstream values and parameter/input gradients in both query modes."""
    torch.manual_seed(23)
    config = QuerySelectorConfig(
        hidden_dim=8, num_classes=5, num_queries=12, learned_content=learned
    )
    selector = QuerySelector(config).train(training)
    # Nonzero box deltas are essential: a zero final layer can hide broken gradients.
    assert isinstance(selector.enc_bbox_head, BoxHead)
    output_layer = selector.enc_bbox_head.layers[-1]
    assert isinstance(output_layer, nn.Linear)
    nn.init.normal_(output_layer.weight, std=0.1)
    nn.init.normal_(output_layer.bias, std=0.1)
    oracle = stage_oracle()
    for name, module in selector.named_children():
        setattr(oracle, name, copy.deepcopy(module))
    oracle.eps = config.anchor_eps
    oracle.eval_spatial_size = None
    oracle.num_queries = config.num_queries
    oracle.learnt_init_query = learned
    oracle.train(training)

    features = encoded()
    oracle_memory = features.memory.detach().clone().requires_grad_()
    target, refs, boxes, logits = oracle._get_decoder_input(
        oracle_memory, features.spatial_shapes.tolist()
    )
    actual = selector(features)
    for ours, theirs in (
        (actual.queries.content, target),
        (actual.queries.ref_logits.detach(), refs),
        (actual.encoder_boxes, boxes),
        (actual.encoder_logits, logits),
    ):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)
    # Recover upstream top-k directly from its unchanged encoder heads.
    _, valid = oracle._generate_anchors(features.spatial_shapes.tolist())
    upstream_logits = oracle.enc_score_head(oracle.enc_output(oracle_memory * valid.float()))
    upstream_indices = upstream_logits.max(-1).values.topk(config.num_queries, dim=1).indices
    assert torch.equal(actual.queries.source_indices, upstream_indices)
    assert actual.queries.content.requires_grad == learned
    assert actual.queries.ref_logits.requires_grad  # assembly performs the eventual detach
    assert not refs.requires_grad

    loss = actual.encoder_logits.square().sum() + actual.encoder_boxes.square().sum()
    oracle_loss = logits.square().sum() + boxes.square().sum()
    if learned:
        loss = loss + actual.queries.content.square().sum()
        oracle_loss = oracle_loss + target.square().sum()
    loss.backward()
    oracle_loss.backward()
    torch.testing.assert_close(features.memory.grad, oracle_memory.grad, rtol=1e-5, atol=1e-6)
    expected_parameters = dict(oracle.named_parameters())
    for name, parameter in selector.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, expected_parameters[name].grad, rtol=1e-5, atol=1e-6
        )


def test_custom_heads_are_registered_receive_gradients_and_keep_initialization() -> None:
    """Keep injected heads' parameters, initialization, and gradient participation."""
    score = nn.Linear(8, 3)
    box = nn.Linear(8, 4)
    weight = score.weight.detach().clone()
    selector = QuerySelector(
        QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=4),
        score_head=score,
        box_head=box,
    )
    assert selector.enc_score_head is score
    assert selector.enc_bbox_head is box
    assert torch.equal(weight, score.weight)
    output = selector(encoded())
    (output.encoder_logits.sum() + output.encoder_boxes.sum()).backward()
    assert score.weight.grad is not None
    assert box.weight.grad is not None
    assert "enc_score_head.weight" in selector.state_dict()
    parameters = list(selector.parameters())
    assert len(parameters) == len({id(parameter) for parameter in parameters})


def test_selector_rejects_incompatible_head_output() -> None:
    """Reject a distribution-shaped head at the coordinate-delta boundary."""
    selector = QuerySelector(
        QuerySelectorConfig(hidden_dim=8, num_classes=3, num_queries=4),
        box_head=nn.Linear(8, 16),
    )
    with pytest.raises(ValueError, match="coordinate deltas"):
        selector(encoded())


def test_invalid_anchors_mask_memory_but_are_not_removed_from_topk() -> None:
    """Keep upstream score-selection behavior even when an invalid anchor wins."""
    selector = QuerySelector(QuerySelectorConfig(hidden_dim=2, num_classes=1, num_queries=1))
    projection = selector.enc_output[0]
    score_head = selector.enc_score_head
    assert isinstance(projection, nn.Linear)
    assert isinstance(score_head, nn.Linear)
    assert projection.bias is not None
    assert score_head.bias is not None
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
        projection.bias.copy_(torch.tensor([1.0, 0.0]))
        score_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        score_head.bias.zero_()
    features = encoded(shapes=((80, 3),), channels=2, batch=1)
    with torch.no_grad():
        features.memory.copy_(torch.tensor([-1.0, 1.0]))
    output = selector(features)
    _, valid = generate_anchors(((80, 3),))
    index = int(output.queries.source_indices[0, 0])
    assert not valid[0, index, 0]
    assert torch.isposinf(output.queries.ref_logits).all()
    assert torch.equal(output.encoder_boxes, torch.ones_like(output.encoder_boxes))


def test_selector_rejects_too_few_positions() -> None:
    """Fail before top-k when the memory cannot supply the requested queries."""
    selector = QuerySelector(QuerySelectorConfig(hidden_dim=8, num_queries=300))
    with pytest.raises(ValueError, match="at least num_queries"):
        selector(encoded())


def test_invalid_metadata_is_rejected_before_selection() -> None:
    """Reject inconsistent offsets, shape dtype, and sequence length."""
    features = encoded()
    with pytest.raises(ValueError, match="prefix sum"):
        replace(features, level_start_index=torch.tensor([0, 1, 2])).validate()
    with pytest.raises((ValueError, TypeCheckError), match=r"(?:int64|Int64)"):
        replace(features, spatial_shapes=features.spatial_shapes.float()).validate()
    with pytest.raises(ValueError, match="sequence length"):
        replace(features, memory=features.memory[:, :-1]).validate()


def test_feature_spec_mismatch_is_rejected() -> None:
    """Reject mismatched input specs and non-increasing construction strides."""
    projector = MemoryProjector((FeatureSpec(3, 8),), hidden_dim=8)
    with pytest.raises(ValueError, match="FeatureSpec"):
        projector(MultiScaleFeatures((torch.randn(2, 3, 5, 7),), (FeatureSpec(3, 16),)))
    with pytest.raises(ValueError, match="strictly increasing"):
        MemoryProjector((FeatureSpec(3, 16), FeatureSpec(5, 8)))


def test_inverse_sigmoid_endpoint_convention_and_gradient() -> None:
    """Preserve separately clamped endpoints and interior inverse-sigmoid gradients."""
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], requires_grad=True)
    logits = inverse_sigmoid(values)
    expected = torch.tensor([1e-5, 1 / 3, 1.0, 3.0, 1e5]).log()
    torch.testing.assert_close(logits, expected)
    logits[1:4].sum().backward()
    assert values.grad is not None
    torch.testing.assert_close(values.grad[1:4], 1 / (values[1:4] * (1 - values[1:4])))
