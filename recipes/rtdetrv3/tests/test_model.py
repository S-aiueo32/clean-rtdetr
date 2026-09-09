# Copyright 2026 Clean RT-DETR contributors
"""Integration contracts for the RT-DETRv3 training branches."""
# ruff: noqa: PLR2004 - explicit tensor dimensions in regression tests

import pytest
import torch
from lightning.pytorch import Trainer

from clean_rtdetr.contracts import DecoderInput
from clean_rtdetr.models.backbone import PResNet
from clean_rtdetr.models.decoder import Decoder, DecoderConfig
from clean_rtdetr.models.encoder import HybridEncoder, HybridEncoderConfig
from clean_rtdetr.models.memory import MemoryProjector
from clean_rtdetr.models.queries import QueryAssembler
from clean_rtdetr.models.rtdetr import RTDETR
from clean_rtdetr.models.selector import QuerySelector, QuerySelectorConfig
from clean_rtdetr_training import DetectionTarget
from clean_rtdetr_training.lightning import RTDETRLightningModule
from rtdetrv3_recipe.criterion import RTDETRv3Criterion
from rtdetrv3_recipe.model import RTDETRv3TrainingModel, perturb_queries


def _tiny() -> RTDETRv3TrainingModel:
    """Build real detector stages with small decoder dimensions."""
    torch.set_num_threads(1)
    backbone = PResNet(depth=18, freeze_at=-1, freeze_norm=False)
    encoder = HybridEncoder(
        backbone.specs,
        HybridEncoderConfig(hidden_dim=16, num_heads=4, feedforward_dim=32),
    )
    base = RTDETR(
        backbone,
        encoder,
        MemoryProjector(encoder.output_specs, hidden_dim=16),
        QuerySelector(QuerySelectorConfig(hidden_dim=16, num_classes=1, num_queries=4)),
        QueryAssembler(),
        Decoder(DecoderConfig(hidden_dim=16, num_classes=1, num_layers=2, num_heads=4)),
    )
    return RTDETRv3TrainingModel(base, sap_groups=2, o2m_queries=8, num_denoising=4)


def _targets(batch: int = 2) -> list[DetectionTarget]:
    """Build person annotations for the training tests."""
    target = DetectionTarget(
        torch.tensor([0], dtype=torch.int64), torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    )
    return [target for _ in range(batch)]


def test_training_branches_share_gradients_and_support_empty_targets() -> None:
    """All training branches contribute gradients and support empty annotations."""
    model = _tiny().train()
    images = torch.randn(2, 3, 64, 64)
    output = model(images, targets=_targets())
    assert len(output.sap) == 2
    assert output.o2m is not None
    assert output.cnn is not None
    assert output.main.denoising is not None
    assert all(group.denoising is not None for group in output.sap)
    assert output.o2m.denoising is None
    losses = RTDETRv3Criterion()(output, _targets())
    loss = torch.stack(list(losses.values())).sum()
    assert bool(torch.isfinite(loss))
    assert loss.item() != 0
    loss.backward()
    for name in (
        "model.decoder",
        "model.selector",
        "sap_selectors.0",
        "sap_selectors.1",
        "o2m_selector",
        "map_memory",
        "cnn",
        "denoising",
    ):
        gradients = [
            parameter.grad for key, parameter in model.named_parameters() if key.startswith(name)
        ]
        assert gradients
        assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
        assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    empty = [DetectionTarget(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))] * 2
    empty_output = model(images, targets=empty)
    assert empty_output.main.denoising is None
    empty_losses = RTDETRv3Criterion()(empty_output, empty)
    empty_total = torch.stack(list(empty_losses.values())).sum()
    assert bool(torch.isfinite(empty_total))
    empty_total.backward()


def test_predict_matches_baseline_and_restores_modes() -> None:
    """Inference skips training branches and restores every module's mode."""
    model = _tiny().train()
    model.sap_selectors[0].eval()
    images = torch.randn(2, 3, 64, 64)
    model.model.eval()
    expected = model.model(images)
    mode_before = {id(module): module.training for module in model.modules()}
    calls = dict.fromkeys(("sap", "o2m", "map_memory", "cnn"), 0)
    hooks = [
        selector.register_forward_hook(lambda *_: calls.__setitem__("sap", calls["sap"] + 1))
        for selector in model.sap_selectors
    ]
    hooks.extend(
        [
            model.o2m_selector.register_forward_hook(
                lambda *_: calls.__setitem__("o2m", calls["o2m"] + 1)
            ),
            model.map_memory.register_forward_hook(
                lambda *_: calls.__setitem__("map_memory", calls["map_memory"] + 1)
            ),
            model.cnn.register_forward_hook(lambda *_: calls.__setitem__("cnn", calls["cnn"] + 1)),
        ]
    )
    predicted = model.predict(images)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(predicted.final.logits, expected.final.logits)
    torch.testing.assert_close(predicted.final.boxes, expected.final.boxes)
    assert model.training
    assert not model.sap_selectors[0].training
    assert mode_before == {id(module): module.training for module in model.modules()}
    assert calls == dict.fromkeys(calls, 0)
    assert predicted.sap == ()
    assert predicted.o2m is None
    assert predicted.cnn is None


def test_perturb_queries_is_reproducible_and_preserves_dn_mask() -> None:
    """SAP replays from RNG state and only modifies the matching suffix."""
    base_mask = torch.zeros(7, 7, dtype=torch.bool)
    base_mask[0, 1] = True
    queries = DecoderInput(torch.randn(2, 7, 16), torch.randn(2, 7, 4), base_mask, 4)
    torch.manual_seed(9)
    first = perturb_queries(queries, 0.5)
    torch.manual_seed(9)
    second = perturb_queries(queries, 0.5)
    assert first.attention_mask is not None
    assert second.attention_mask is not None
    first_mask = first.attention_mask
    second_mask = second.attention_mask
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_mask[:3], base_mask[:3])
    assert torch.equal(first_mask[:, :3], base_mask[:, :3])
    assert not first_mask[-4:, -4:].all(dim=-1).any()


def test_invalid_branch_configuration_is_rejected() -> None:
    """Reject invalid counts and probabilities before allocating branches."""
    model = _tiny().model
    with pytest.raises(ValueError, match="sap_groups"):
        RTDETRv3TrainingModel(model, sap_groups=-1)
    with pytest.raises(ValueError, match="o2m_queries"):
        RTDETRv3TrainingModel(model, o2m_queries=0)
    with pytest.raises(ValueError, match="SAP"):
        RTDETRv3TrainingModel(model, mask_probability=1)


def test_checkpoint_configuration_records_resume_sensitive_changes() -> None:
    """Same-shaped models record settings that change their training trajectory."""
    first = _tiny()
    second = RTDETRv3TrainingModel(
        first.model, sap_groups=2, mask_probability=0.25, o2m_queries=8, num_denoising=6
    )
    assert (
        first.checkpoint_config()["mask_probability"]
        != second.checkpoint_config()["mask_probability"]
    )
    assert first.checkpoint_config()["denoising"] != second.checkpoint_config()["denoising"]
    assert first.checkpoint_config()["sap_selectors"] == second.checkpoint_config()["sap_selectors"]


def test_lightning_checkpoint_rejects_changed_recipe_configuration() -> None:
    """The Lightning checkpoint boundary rejects changes invisible in state_dict."""
    model = _tiny()
    criterion = RTDETRv3Criterion()
    module = RTDETRLightningModule(model, criterion, torch.optim.AdamW)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0,
    )
    trainer.fit(module, train_dataloaders=[(torch.randn(1, 3, 64, 64), _targets(1))])
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    criterion.repeats = 5
    with pytest.raises(ValueError, match="recipe"):
        module.on_load_checkpoint(checkpoint)
