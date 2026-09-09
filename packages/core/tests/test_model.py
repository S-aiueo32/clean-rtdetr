# Copyright 2026 Clean RT-DETR contributors
"""Integration checks for the explicit image-to-prediction composition."""

from dataclasses import replace

import pytest
import torch
from core_model_fixtures import small_model

from clean_rtdetr import (
    RTDETR,
    Decoder,
    DecoderPlan,
    FeatureSpec,
    HybridEncoder,
    HybridEncoderConfig,
    ModelFactory,
    create_model,
    get_model_preset,
    list_models,
    rtdetr_r50vd,
)

BASELINE_CLASSES = 80
BASELINE_QUERIES = 300
BASELINE_LAYERS = 6
BASELINE_HIDDEN = 256
BASELINE_LEVELS = 3
CUSTOM_CLASSES = 7
MODEL_NAMES = (
    "rtdetr_r18vd",
    "rtdetr_r34vd",
    "rtdetr_r50vd_m",
    "rtdetr_r50vd",
    "rtdetr_r101vd",
    "rtdetrv2_r18vd",
    "rtdetrv2_r34vd",
    "rtdetrv2_r50vd_m",
    "rtdetrv2_r50vd",
    "rtdetrv2_r101vd",
)
MODEL_ARTIFACTS = {
    "rtdetr_r18vd": ("rtdetr_r18vd_6x_coco.yml", "v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
    "rtdetr_r34vd": ("rtdetr_r34vd_6x_coco.yml", "v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
    "rtdetr_r50vd_m": ("rtdetr_r50vd_m_6x_coco.yml", "v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
    "rtdetr_r50vd": ("rtdetr_r50vd_6x_coco.yml", "v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"),
    "rtdetr_r101vd": ("rtdetr_r101vd_6x_coco.yml", "v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"),
    "rtdetrv2_r18vd": (
        "rtdetrv2_r18vd_120e_coco.yml",
        "v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
    ),
    "rtdetrv2_r34vd": ("rtdetrv2_r34vd_120e_coco.yml", "v0.1/rtdetrv2_r34vd_120e_coco_ema.pth"),
    "rtdetrv2_r50vd_m": ("rtdetrv2_r50vd_m_7x_coco.yml", "v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth"),
    "rtdetrv2_r50vd": ("rtdetrv2_r50vd_6x_coco.yml", "v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"),
    "rtdetrv2_r101vd": (
        "rtdetrv2_r101vd_6x_coco.yml",
        "v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
    ),
}


def test_model_matches_manual_stage_composition() -> None:
    """The wrapper must preserve the outputs of its injected stages exactly."""
    torch.manual_seed(5)
    model = small_model().eval()
    images = torch.randn(1, 3, 64, 64)
    actual = model(images, plan=DecoderPlan(num_layers=1))
    features = model.backbone(images)
    encoded_features = model.encoder(features)
    encoded = model.projector(encoded_features.features)
    selected = model.selector(encoded)
    queries = model.assembler(selected.queries)
    expected = model.decoder(queries, encoded, plan=DecoderPlan(num_layers=1))
    torch.testing.assert_close(actual.decoder.logits, expected.logits)
    torch.testing.assert_close(actual.decoder.boxes, expected.boxes)
    torch.testing.assert_close(actual.encoder_aux.logits, selected.encoder_logits)


def test_model_validates_component_contracts() -> None:
    """Reject mismatched metadata and dimensions during construction."""
    model = small_model()
    with pytest.raises(ValueError, match="feature specs"):
        RTDETR(
            model.backbone,
            HybridEncoder(
                (FeatureSpec(512, 8), FeatureSpec(1024, 16), FeatureSpec(1024, 32)),
                HybridEncoderConfig(hidden_dim=8, num_heads=2, feedforward_dim=16),
            ),
            model.projector,
            model.selector,
            model.assembler,
            model.decoder,
        )


def test_model_forwards_plan_and_aifi_without_changing_predictions() -> None:
    """AIFI capture is observational and decoder plans select layer prefixes."""
    torch.manual_seed(9)
    model = small_model().eval()
    images = torch.randn(1, 3, 64, 64)
    plain = model(images, plan=DecoderPlan(num_layers=2), collect_aifi=False)
    tapped = model(images, plan=DecoderPlan(num_layers=2), collect_aifi=True)
    assert tapped.after_aifi is not None
    torch.testing.assert_close(plain.decoder.logits, tapped.decoder.logits)
    assert model(images, plan=DecoderPlan(num_layers=1)).decoder.logits.shape[0] == 1
    with pytest.raises(ValueError, match="num_layers"):
        model(images, plan=DecoderPlan(num_layers=0))


def test_model_rejects_invalid_image_before_downstream_work() -> None:
    """Preserve the backbone input contract at the public model boundary."""
    model = small_model()
    with pytest.raises(ValueError, match="multiples of 32"):
        model(torch.randn(1, 3, 63, 64))


@pytest.mark.parametrize(
    ("kind", "match"),
    [("hidden", "hidden dimensions"), ("classes", "class counts"), ("levels", "level count")],
)
def test_model_rejects_each_decoder_contract_mismatch(kind: str, match: str) -> None:
    """Validate decoder hidden size, class count, and feature level count independently."""
    model = small_model()
    changes = {
        "hidden": {"hidden_dim": 16},
        "classes": {"num_classes": 4},
        "levels": {"num_levels": 2},
    }
    replacement = Decoder(replace(model.decoder.config, **changes[kind]))
    with pytest.raises(ValueError, match=match):
        RTDETR(
            model.backbone,
            model.encoder,
            model.projector,
            model.selector,
            model.assembler,
            replacement,
        )


@pytest.mark.parametrize("loss_source", ["encoder_aux", "aifi"])
def test_auxiliary_and_aifi_losses_retain_gradient_routes(loss_source: str) -> None:
    """Auxiliary predictions and captured AIFI features remain connected to the model."""
    torch.manual_seed(12)
    model = small_model().train()
    images = torch.randn(1, 3, 64, 64, requires_grad=True)
    output = model(images, collect_aifi=True)
    assert output.after_aifi is not None
    loss = (
        output.encoder_aux.logits.square().mean() + output.encoder_aux.boxes.square().mean()
        if loss_source == "encoder_aux"
        else output.after_aifi.levels[0].square().mean()
    )
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    for module in (model.encoder, model.backbone):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_decoder_loss_does_not_backpropagate_through_assembled_references() -> None:
    """The query assembly boundary detaches selector reference logits from decoding."""
    model = small_model().train()
    images = torch.randn(1, 3, 64, 64)
    features = model.backbone(images)
    encoded_features = model.encoder(features)
    encoded = model.projector(encoded_features.features)
    selected = model.selector(encoded)
    selected.queries.ref_logits.retain_grad()
    queries = model.assembler(selected.queries)
    model.decoder(queries, encoded).logits.sum().backward()
    assert selected.queries.ref_logits.grad is None


def test_r50vd_factory_uses_pinned_baseline_dimensions() -> None:
    """The public factory exposes the planned baseline dimensions."""
    model = rtdetr_r50vd()
    assert model.selector.config.num_classes == BASELINE_CLASSES
    assert model.selector.config.num_queries == BASELINE_QUERIES
    assert model.selector.config.learned_content is False
    assert model.decoder.config.num_layers == BASELINE_LAYERS
    assert model.decoder.config.hidden_dim == BASELINE_HIDDEN
    assert model.decoder.config.num_levels == BASELINE_LEVELS


def test_named_factory_exposes_all_official_presets() -> None:
    """The generic entry point preserves upstream hub model ordering."""
    assert list_models() == MODEL_NAMES


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_named_factory_builds_each_official_architecture(name: str) -> None:
    """Every advertised preset constructs its resolved stage dimensions."""
    preset = get_model_preset(name)
    with torch.device("meta"):
        model = create_model(name, num_classes=CUSTOM_CLASSES)
    assert model.backbone.depth == preset.backbone_depth
    assert model.encoder.hidden_dim == preset.hidden_dim
    assert model.encoder.config.expansion == preset.encoder_expansion
    assert model.selector.num_classes == CUSTOM_CLASSES
    assert model.decoder.config.num_layers == preset.decoder_layers
    assert model.decoder.default_eval_plan == (preset.default_eval_plan or DecoderPlan())
    point_scale_key = "decoder.layers.0.cross_attn.num_points_scale"
    assert (point_scale_key in model.state_dict()) is (preset.family == "rtdetrv2")
    residual_stabilizer = model.decoder.get_submodule("layers.0.residual_stabilizer")
    assert isinstance(residual_stabilizer, torch.nn.Identity) is (preset.family == "rtdetrv2")


def test_model_presets_retain_official_source_and_checkpoint_urls() -> None:
    """Factory metadata points to each selected upstream YAML and weight artifact."""
    for name in MODEL_NAMES:
        preset = get_model_preset(name)
        assert preset.config_url.startswith(
            "https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/"
        )
        config, checkpoint = MODEL_ARTIFACTS[name]
        assert preset.config_url.endswith(config)
        assert preset.checkpoint_url.startswith(
            "https://github.com/lyuwenyu/storage/releases/download/"
        )
        assert preset.checkpoint_url.endswith(checkpoint)


def test_model_factory_requires_matching_preset_and_builder_names() -> None:
    """A custom factory cannot advertise a preset without a builder."""
    with pytest.raises(ValueError, match="identical names"):
        ModelFactory({}, {"extra": lambda _classes: small_model()})


def test_model_factory_reports_unknown_name_and_available_models() -> None:
    """Unknown names fail before construction and advertise valid choices."""
    with pytest.raises(ValueError, match=r"Unknown model 'missing'.*rtdetr_r50vd"):
        create_model("missing")
