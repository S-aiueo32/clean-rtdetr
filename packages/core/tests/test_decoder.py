# Copyright 2026 Clean RT-DETR contributors
"""Verify query assembly and decoder parity with the pinned upstream classes."""

import copy

import pytest
import torch
from oracle import decoder_oracle
from torch import nn

from clean_rtdetr import (
    Decoder,
    DecoderConfig,
    DecoderPlan,
    DenoisingQueries,
    QueryAssembler,
    QuerySet,
)
from clean_rtdetr.contracts import EncodedFeatures
from clean_rtdetr.models.attention import MultiScaleDeformableAttention, deformable_attention_core
from clean_rtdetr.models.selector import BoxHead

DEFAULT_EVAL_LAYERS = 2


def decoder_config() -> DecoderConfig:
    """Return a small multi-level decoder configuration for CPU parity tests."""
    return DecoderConfig(
        hidden_dim=8,
        num_classes=5,
        num_layers=3,
        num_heads=2,
        num_levels=2,
        num_points=2,
        feedforward_dim=16,
    )


def encoded(*, requires_grad: bool = False) -> EncodedFeatures:
    """Create non-square memory and exact prefix-sum metadata."""
    shapes = torch.tensor(((3, 4), (2, 3)), dtype=torch.int64)
    sizes = shapes.prod(1)
    memory = torch.randn(2, int(sizes.sum()), 8, requires_grad=requires_grad)
    return EncodedFeatures(memory, shapes, sizes.cumsum(0) - sizes)


def query_set(*, requires_grad: bool = False) -> QuerySet:
    """Create matching queries whose references expose the assembly detach route."""
    return QuerySet(
        content=torch.randn(2, 5, 8, requires_grad=requires_grad),
        ref_logits=torch.randn(2, 5, 4, requires_grad=True),
        source_indices=torch.arange(5, dtype=torch.int64).expand(2, -1),
    )


def test_query_assembly_prepends_denoising_and_detaches_all_references() -> None:
    """Preserve DN-first ordering, mask identity, content gradients, and reference detach."""
    matching = query_set(requires_grad=True)
    denoising_content = torch.randn(2, 3, 8, requires_grad=True)
    denoising_refs = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    result = QueryAssembler()(
        matching,
        DenoisingQueries(denoising_content, denoising_refs, mask),
    )

    matching_query_count = matching.content.shape[1]
    assert result.matching_queries == matching_query_count
    assert result.attention_mask is mask
    assert not result.ref_logits.requires_grad
    torch.testing.assert_close(result.content[:, :3], denoising_content)
    torch.testing.assert_close(result.content[:, 3:], matching.content)
    torch.testing.assert_close(result.ref_logits[:, :3], denoising_refs.detach())
    result.content.sum().backward()
    assert denoising_content.grad is not None
    assert matching.content.grad is not None
    assert denoising_refs.grad is None
    assert matching.ref_logits.grad is None


def test_deformable_attention_core_samples_level_pixel_centers() -> None:
    """Fix normalized coordinates, level splitting, and align_corners=False sampling."""
    value = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[10.0]]]])
    shapes = torch.tensor(((2, 2), (1, 1)), dtype=torch.int64)
    locations = torch.tensor(((0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.5, 0.5))).reshape(
        1, 2, 1, 2, 1, 2
    )
    weights = torch.tensor((0.25, 0.75, 0.5, 0.5)).reshape(1, 2, 1, 2, 1)

    result = deformable_attention_core(value, shapes, locations, weights)

    torch.testing.assert_close(result, torch.tensor([[[7.75], [7.0]]]))


def test_v2_standard_attention_registers_scale_and_matches_v1() -> None:
    """Use the official v2 scale buffer with equivalent four-point sampling."""
    torch.manual_seed(37)
    v1 = MultiScaleDeformableAttention(8, 2, 2, 4)
    v2 = MultiScaleDeformableAttention(8, 2, 2, 4, point_scale=torch.full((8,), 0.25))
    assert isinstance(v1.value_proj, nn.Linear)
    assert isinstance(v1.output_proj, nn.Linear)
    v2.load_state_dict(v1.state_dict(), strict=False)
    assert "num_points_scale" not in v1.state_dict()
    assert "value_proj.weight" in v2.state_dict()
    assert "output_proj.weight" in v2.state_dict()
    torch.testing.assert_close(v2.num_points_scale, torch.full((8,), 0.25))
    query = torch.randn(2, 5, 8)
    features = encoded()
    references = torch.rand(2, 5, 2, 4)
    torch.testing.assert_close(
        v2(query, references, features.memory, features.spatial_shapes),
        v1(query, references, features.memory, features.spatial_shapes),
    )


@pytest.mark.parametrize("training", [False, True])
def test_decoder_outputs_and_gradients_match_pinned_upstream(*, training: bool) -> None:
    """Match layer computations, refinement values, and the training detach gradient route."""
    torch.manual_seed(41)
    config = decoder_config()
    decoder = Decoder(config).train(training)
    # Nonzero deltas are required to exercise iterative refinement gradients.
    for index, head in enumerate(decoder.dec_bbox_head):
        assert isinstance(head, BoxHead)
        output_layer = head.layers[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.normal_(output_layer.weight, std=0.05 * (index + 1))
        nn.init.normal_(output_layer.bias, std=0.05)

    executed_layers = config.num_layers if training else 2
    oracle = decoder_oracle(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        activation=config.activation,
        num_levels=config.num_levels,
        num_points=config.num_points,
        num_layers=config.num_layers,
        eval_index=executed_layers - 1,
    ).train(training)
    oracle.load_state_dict(
        {
            name: parameter
            for name, parameter in decoder.state_dict().items()
            if name.startswith("layers.")
        }
    )
    oracle_box_heads = copy.deepcopy(decoder.dec_bbox_head)
    oracle_score_heads = copy.deepcopy(decoder.dec_score_head)
    oracle_position_head = copy.deepcopy(decoder.query_pos_head)

    features = encoded(requires_grad=True)
    matching = query_set(requires_grad=True)
    assembled = QueryAssembler()(matching)
    oracle_memory = features.memory.detach().clone().requires_grad_()
    oracle_content = assembled.content.detach().clone().requires_grad_()
    oracle_refs = assembled.ref_logits.detach().clone()
    expected_boxes, expected_logits = oracle(
        oracle_content,
        oracle_refs,
        oracle_memory,
        features.spatial_shapes.tolist(),
        features.level_start_index.tolist(),
        oracle_box_heads,
        oracle_score_heads,
        oracle_position_head,
    )
    actual = decoder(
        assembled,
        features,
        plan=DecoderPlan(
            num_layers=executed_layers,
            collect_intermediate=training,
        ),
    )

    torch.testing.assert_close(actual.boxes, expected_boxes, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.logits, expected_logits, rtol=1e-5, atol=1e-6)
    actual_loss = actual.boxes.square().sum() + actual.logits.square().sum()
    expected_loss = expected_boxes.square().sum() + expected_logits.square().sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(features.memory.grad, oracle_memory.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(matching.content.grad, oracle_content.grad, rtol=1e-5, atol=1e-6)
    assert matching.ref_logits.grad is None

    oracle_parameters = {
        **dict(oracle.named_parameters()),
        **{f"dec_bbox_head.{name}": value for name, value in oracle_box_heads.named_parameters()},
        **{
            f"dec_score_head.{name}": value for name, value in oracle_score_heads.named_parameters()
        },
        **{
            f"query_pos_head.{name}": value
            for name, value in oracle_position_head.named_parameters()
        },
    }
    for name, parameter in decoder.named_parameters():
        expected_gradient = oracle_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            rtol=1e-5,
            atol=1e-6,
        )


def test_decoder_plan_selects_prefix_and_output_collection_explicitly() -> None:
    """Return all executed stages or only the final stage without relying on eval mode."""
    decoder = Decoder(decoder_config()).eval()
    inputs = QueryAssembler()(query_set())
    features = encoded()

    all_outputs = decoder(
        inputs,
        features,
        plan=DecoderPlan(num_layers=2, collect_intermediate=True),
    )
    final_only = decoder(
        inputs,
        features,
        plan=DecoderPlan(num_layers=2, collect_intermediate=False),
    )

    executed_layers = 2
    assert all_outputs.logits.shape[0] == executed_layers
    assert final_only.logits.shape[0] == 1
    torch.testing.assert_close(final_only.final.logits, all_outputs.final.logits)
    torch.testing.assert_close(final_only.final.boxes, all_outputs.final.boxes)
    with pytest.raises(ValueError, match="num_layers"):
        decoder(inputs, features, plan=DecoderPlan(num_layers=0))


def test_injected_eval_plan_controls_default_depth_but_explicit_depth_takes_precedence() -> None:
    """Apply the injected eval depth while preserving explicit execution plans."""
    config = decoder_config()
    decoder = Decoder(
        config,
        default_eval_plan=DecoderPlan(
            num_layers=DEFAULT_EVAL_LAYERS,
            collect_intermediate=False,
        ),
    ).eval()
    inputs = QueryAssembler()(query_set())
    features = encoded()
    calls = [0, 0, 0]
    hooks = [
        layer.register_forward_hook(
            lambda _module, _args, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, layer in enumerate(decoder.layers)
    ]
    default = decoder(inputs, features)
    assert calls == [1, 1, 0]
    assert default.logits.shape[0] == 1
    collected = decoder(inputs, features, plan=DecoderPlan(collect_intermediate=True))
    assert calls == [2, 2, 0]
    assert collected.logits.shape[0] == DEFAULT_EVAL_LAYERS
    explicit_depth = config.num_layers
    explicit = decoder(
        inputs,
        features,
        plan=DecoderPlan(num_layers=explicit_depth, collect_intermediate=True),
    )
    assert explicit.logits.shape[0] == explicit_depth
    decoder.train()
    trained = decoder(inputs, features)
    assert trained.logits.shape[0] == config.num_layers
    for hook in hooks:
        hook.remove()


def test_decoder_rejects_mismatched_feature_levels() -> None:
    """Reject metadata that cannot satisfy the configured deformable attention levels."""
    config = decoder_config()
    decoder = Decoder(config)
    shapes = torch.tensor(((3, 4),), dtype=torch.int64)
    features = EncodedFeatures(torch.randn(2, 12, 8), shapes, torch.tensor((0,)))
    with pytest.raises(ValueError, match="feature levels"):
        decoder(QueryAssembler()(query_set()), features)
