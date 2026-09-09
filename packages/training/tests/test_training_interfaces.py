# Copyright 2026 Clean RT-DETR contributors
"""The Lightning adapter depends only on the public training interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, assert_type, cast

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

lightning = pytest.importorskip("lightning")
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

from clean_rtdetr.contracts import (  # noqa: E402
    DecoderInput,
    DecoderOutput,
    DenoisingQueries,
    DetectionOutput,
    ModelOutput,
    QuerySet,
)
from clean_rtdetr.models.interfaces import Assembler, Detector  # noqa: E402
from clean_rtdetr.models.queries import QueryAssembler  # noqa: E402
from clean_rtdetr_training import DenoisingBatch as PublicDenoisingBatch  # noqa: E402
from clean_rtdetr_training import DenoisingTrainingModel, SetCriterion  # noqa: E402
from clean_rtdetr_training.contracts import (  # noqa: E402
    DenoisingBatch,
    DenoisingMetadata,
    DetectionTarget,
    MatchIndices,
    TrainingOutput,
    validate_detection,
)
from clean_rtdetr_training.interfaces import (  # noqa: E402
    DenoisingGenerator,
    Matcher,
    TrainingCriterion,
    TrainingModel,
)
from clean_rtdetr_training.lightning import RTDETRLightningModule  # noqa: E402
from clean_rtdetr_training.models import DenoisingBatch as ModelsDenoisingBatch  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from clean_rtdetr.models.decoder import DecoderPlan


@dataclass(frozen=True)
class ToyPrediction:
    """An intentionally non-RT-DETR prediction payload."""

    values: Tensor


class ToyTrainingModel(TrainingModel[ToyPrediction]):
    """Minimal trainable implementation of the model interface."""

    def __init__(self, *, recipe: str = "toy") -> None:
        """Create a linear model with a checkpointed recipe selector."""
        super().__init__()
        self.projection = nn.Linear(3, 1, bias=False)
        self.recipe = recipe
        self.predict_calls = 0

    def forward(
        self,
        images: Tensor,
        *,
        targets: Sequence[DetectionTarget] | None = None,
    ) -> ToyPrediction:
        """Produce a payload through the standard module call path."""
        del targets
        return ToyPrediction(self.projection(images.mean(dim=(-2, -1))).squeeze(-1))

    def predict(self, images: Tensor) -> ToyPrediction:
        """Record and dispatch target-free inference."""
        self.predict_calls += 1
        return self(images)

    def checkpoint_config(self) -> dict[str, object]:
        """Return the non-tensor model recipe."""
        return {"recipe": self.recipe}


class AlternateToyTrainingModel(ToyTrainingModel):
    """Same state and config, but a different selectable implementation."""


class ToyCriterion(TrainingCriterion[ToyPrediction]):
    """Criterion with its own trainable parameter."""

    def __init__(
        self,
        *,
        recipe: str = "mse",
        shared: nn.Parameter | None = None,
        matcher: Matcher | None = None,
    ) -> None:
        """Create a loss scale, optionally shared with the model."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0)) if shared is None else shared
        self.recipe = recipe
        self.matcher = matcher

    def forward(
        self, output: ToyPrediction, targets: Sequence[DetectionTarget]
    ) -> dict[str, Tensor]:
        """Compute a scalar loss from the arbitrary prediction payload."""
        if self.matcher is not None:
            logits = output.values[:, None, None].expand(-1, 1, 3)
            boxes = output.values.sigmoid()[:, None, None].expand(-1, 1, 4)
            self.matcher(DetectionOutput(logits, boxes), targets)
        expected = torch.stack([target.boxes.mean() for target in targets])
        return {"toy": (output.values - expected).square().mean() * self.scale.square()}

    def checkpoint_config(self) -> dict[str, object]:
        """Return the non-tensor criterion recipe."""
        if self.matcher is None:
            return {"recipe": self.recipe}
        matcher_type = type(self.matcher)
        return {
            "recipe": self.recipe,
            "matcher": self.matcher.checkpoint_config(),
            "matcher_class": f"{matcher_type.__module__}.{matcher_type.__qualname__}",
        }


class FixedMatcher(Matcher):
    """Deterministically pair targets with queries without Hungarian internals."""

    def __init__(self, *, reverse: bool = False) -> None:
        """Store a selectable assignment policy and visible module state."""
        super().__init__()
        self.reverse = reverse
        self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

    def forward(
        self, predictions: DetectionOutput, targets: Sequence[DetectionTarget]
    ) -> tuple[MatchIndices, ...]:
        """Pair each target with the first queries, optionally in reverse order."""
        validate_detection(predictions, targets)
        self.call_count.add_(1)
        matches = []
        for target in targets:
            count = target.labels.numel()
            prediction_indices = torch.arange(count, device=predictions.logits.device)
            if self.reverse:
                prediction_indices = prediction_indices.flip(0)
            matches.append(
                MatchIndices(
                    prediction_indices,
                    torch.arange(count, device=predictions.logits.device),
                )
            )
        return tuple(matches)

    @property
    def call_count(self) -> Tensor:
        """Expose the registered invocation counter with a precise type."""
        return self.get_buffer("calls")

    def checkpoint_config(self) -> dict[str, object]:
        """Describe the assignment policy without exposing a config attribute."""
        return {"reverse": self.reverse}


class ToyDetector(Detector):
    """Small detector implementation with no RT-DETR stage attributes."""

    num_classes = 3
    hidden_dim = 2
    num_queries = 1

    def __init__(self) -> None:
        """Create the only projection needed by the toy decoder."""
        super().__init__()
        self.head = nn.Linear(self.hidden_dim, self.num_classes)
        self.box_logit = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        images: Tensor,
        *,
        plan: DecoderPlan | None = None,
        collect_aifi: bool = False,
        denoising: DenoisingQueries | None = None,
    ) -> ModelOutput:
        """Return all DN queries before one learned matching query."""
        del plan, collect_aifi
        batch = images.shape[0]
        matching = images.mean(dim=(1, 2, 3))[:, None, None].expand(
            batch, self.num_queries, self.hidden_dim
        )
        content = matching if denoising is None else torch.cat((denoising.content, matching), dim=1)
        all_logits = self.head(content).unsqueeze(0)
        all_boxes = self.box_logit.sigmoid().expand(batch, content.shape[1], 4).unsqueeze(0)
        decoded = DecoderOutput(all_logits, all_boxes)
        encoder_logits = self.head(matching)
        encoder_boxes = self.box_logit.sigmoid().expand(batch, self.num_queries, 4)
        return ModelOutput(
            decoded,
            DetectionOutput(encoder_logits, encoder_boxes),
            None,
        )

    def checkpoint_config(self) -> dict[str, object]:
        """Describe the detector independently of a stage layout."""
        return {"variant": "toy"}


class ToyAssembler(Assembler):
    """Independent selectable assembler used to exercise stage-level Hydra DI."""

    def __init__(self, *, variant: str = "toy") -> None:
        """Keep a recipe value while delegating the established tensor operation."""
        super().__init__()
        self.variant = variant
        self.implementation = QueryAssembler()

    def forward(
        self, matching: QuerySet, denoising: DenoisingQueries | None = None
    ) -> DecoderInput:
        """Assemble matching and optional denoising queries."""
        return self.implementation(matching, denoising)

    def checkpoint_config(self) -> dict[str, object]:
        """Expose experiment selection without a concrete stage config."""
        return {"variant": self.variant}


class ToyDenoisingGenerator(DenoisingGenerator):
    """Generate fixed-shape learned queries without embedding/config attributes."""

    num_classes = 3
    hidden_dim = 2

    def __init__(self, *, metadata_queries: int = 2) -> None:
        """Create learned content and optionally malformed metadata for boundary tests."""
        super().__init__()
        self.content = nn.Parameter(torch.tensor([0.1, 0.2]))
        self.metadata_queries = metadata_queries

    @property
    def dtype(self) -> torch.dtype:
        """Return the learned query dtype."""
        return self.content.dtype

    @property
    def device(self) -> torch.device:
        """Return the learned query device."""
        return self.content.device

    def forward(
        self, targets: Sequence[DetectionTarget], *, num_queries: int
    ) -> DenoisingBatch | None:
        """Build two DN queries and one positive assignment per image."""
        batch = len(targets)
        count = 2
        queries = DenoisingQueries(
            self.content.expand(batch, count, self.hidden_dim),
            torch.zeros(batch, count, 4, dtype=self.dtype, device=self.device),
            torch.zeros(
                count + num_queries, count + num_queries, dtype=torch.bool, device=self.device
            ),
        )
        metadata = DenoisingMetadata(
            tuple(torch.tensor([0], device=self.device) for _ in targets),
            num_groups=1,
            num_queries=self.metadata_queries,
        )
        return DenoisingBatch(queries, metadata)

    def checkpoint_config(self) -> dict[str, object]:
        """Describe metadata behavior without a config attribute."""
        return {"metadata_queries": self.metadata_queries}


class _IncompleteModel(TrainingModel[ToyPrediction]):
    def forward(
        self,
        images: Tensor,
        *,
        targets: Sequence[DetectionTarget] | None = None,
    ) -> ToyPrediction:
        del targets
        return ToyPrediction(images.mean(dim=(1, 2, 3)))


class _ToyDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        value = float(index + 1) / 4
        return (
            torch.full((3, 16, 16), value),
            DetectionTarget(torch.tensor([0]), torch.full((1, 4), value)),
        )


def _collate(
    samples: Sequence[tuple[Tensor, DetectionTarget]],
) -> tuple[Tensor, list[DetectionTarget]]:
    return torch.stack([image for image, _ in samples]), [target for _, target in samples]


def _loader() -> DataLoader[Any]:
    return DataLoader(_ToyDataset(), batch_size=2, collate_fn=_collate)


def _module(
    *,
    model: ToyTrainingModel | None = None,
    criterion: ToyCriterion | None = None,
) -> RTDETRLightningModule[ToyPrediction]:
    return RTDETRLightningModule(
        model or ToyTrainingModel(),
        criterion or ToyCriterion(),
        partial(torch.optim.SGD, lr=0.1),
    )


def _trainer(
    root: Path, *, epochs: int, checkpoint: ModelCheckpoint | None = None
) -> lightning.Trainer:
    return lightning.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=epochs,
        default_root_dir=root,
        callbacks=[] if checkpoint is None else [checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=checkpoint is not None,
        num_sanity_val_steps=0,
        deterministic=True,
    )


def test_interfaces_are_abstract() -> None:
    """Incomplete implementations cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        _IncompleteModel()  # type: ignore[abstract]


def test_generic_module_dispatches_through_public_interfaces_and_hooks(tmp_path: Path) -> None:
    """Training and prediction use module calls and register both components."""
    model = ToyTrainingModel()
    criterion = ToyCriterion()
    module = _module(model=model, criterion=criterion)
    model_hooks: list[object] = []
    criterion_hooks: list[object] = []
    model.register_forward_hook(lambda *_: model_hooks.append(None))
    criterion.register_forward_hook(lambda *_: criterion_hooks.append(None))
    initial_scale = criterion.scale.detach().clone()

    prediction = module(torch.ones(1, 3, 16, 16))
    assert isinstance(prediction, ToyPrediction)
    assert model.predict_calls == 1
    _trainer(tmp_path, epochs=1).fit(module, train_dataloaders=_loader())

    assert model_hooks
    assert criterion_hooks
    assert dict(module.named_parameters())["criterion.scale"] is criterion.scale
    assert not torch.equal(criterion.scale.detach(), initial_scale)


def test_optimizer_owns_model_and_criterion_parameters_once() -> None:
    """The default optimizer includes criterion state and deduplicates sharing."""
    model = ToyTrainingModel()
    criterion = ToyCriterion(shared=cast("nn.Parameter", model.projection.weight))
    module = _module(model=model, criterion=criterion)
    optimizer: Any = module.configure_optimizers()
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert len(actual) == len({id(parameter) for parameter in actual})
    assert {id(parameter) for parameter in actual} == {id(parameter) for parameter in expected}


def test_checkpoint_restores_generic_criterion_and_rejects_recipe_changes(tmp_path: Path) -> None:
    """Resume restores interface state and validates config and implementation IDs."""
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", save_last=True)
    trained_criterion = ToyCriterion()
    trained = _module(criterion=trained_criterion)
    _trainer(tmp_path / "first", epochs=1, checkpoint=checkpoint).fit(
        trained, train_dataloaders=_loader()
    )
    assert checkpoint.last_model_path
    trained_scale = trained_criterion.scale.detach().clone()

    resumed_criterion = ToyCriterion()
    resumed = _module(criterion=resumed_criterion)
    _trainer(tmp_path / "resume", epochs=1).fit(
        resumed, train_dataloaders=_loader(), ckpt_path=checkpoint.last_model_path
    )
    torch.testing.assert_close(resumed_criterion.scale.detach(), trained_scale, rtol=0, atol=0)

    for incompatible in (
        _module(criterion=ToyCriterion(recipe="absolute")),
        _module(model=AlternateToyTrainingModel()),
    ):
        with pytest.raises(ValueError, match="recipe"):
            _trainer(tmp_path / "incompatible", epochs=2).fit(
                incompatible,
                train_dataloaders=_loader(),
                ckpt_path=checkpoint.last_model_path,
            )


def test_set_criterion_accepts_matcher_interface_and_preserves_module_behavior() -> None:
    """A non-Hungarian matcher participates in hooks, state, losses, and backward."""
    matcher = FixedMatcher()
    criterion = SetCriterion(matcher=matcher)
    hook_calls: list[object] = []
    matcher.register_forward_hook(lambda *_: hook_calls.append(None))
    logits = torch.zeros(2, 2, 3, requires_grad=True)
    boxes = torch.full((2, 2, 4), 0.5, requires_grad=True)
    output = DetectionOutput(logits, boxes)
    targets = [
        DetectionTarget(torch.tensor([0]), torch.tensor([[0.4, 0.4, 0.2, 0.2]])),
        DetectionTarget(torch.tensor([2]), torch.tensor([[0.6, 0.6, 0.3, 0.3]])),
    ]

    losses = criterion.stage_losses(output, targets)
    torch.stack(list(losses.values())).sum().backward()

    assert hook_calls == [None]
    assert matcher.call_count.item() == 1
    assert "matcher.calls" in criterion.state_dict()
    assert logits.grad is not None
    assert boxes.grad is not None


def test_set_criterion_requires_explicit_matcher_selection() -> None:
    """Construction cannot silently select an assignment implementation."""
    criterion_type: Any = SetCriterion
    with pytest.raises(TypeError, match="matcher"):
        criterion_type()


def test_set_criterion_checkpoint_uses_matcher_interface_configuration() -> None:
    """Matcher selection and settings form part of the criterion resume recipe."""
    normal = SetCriterion(matcher=FixedMatcher(reverse=False)).checkpoint_config()
    reversed_ = SetCriterion(matcher=FixedMatcher(reverse=True)).checkpoint_config()

    assert normal["matcher"] == {"reverse": False}
    assert reversed_["matcher"] == {"reverse": True}
    assert normal != reversed_
    assert cast("str", normal["matcher_class"]).endswith(".FixedMatcher")


def test_denoising_wrapper_uses_only_detector_and_generator_interfaces() -> None:
    """Custom modules retain hooks, parameters, gradients, splitting, and provenance."""
    detector = ToyDetector()
    generator = ToyDenoisingGenerator()
    wrapper = DenoisingTrainingModel(detector, generator)
    module = RTDETRLightningModule(
        wrapper,
        SetCriterion(matcher=FixedMatcher()),
        optimizer_factory=partial(torch.optim.SGD, lr=0.01),
    )
    assert_type(module, RTDETRLightningModule[TrainingOutput])
    detector_hooks: list[object] = []
    generator_hooks: list[object] = []
    detector.register_forward_hook(lambda *_: detector_hooks.append(None))
    generator.register_forward_hook(lambda *_: generator_hooks.append(None))
    images = torch.ones(2, 3, 16, 16)
    targets = [
        DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.2]])),
        DetectionTarget(torch.tensor([1]), torch.tensor([[0.4, 0.4, 0.3, 0.3]])),
    ]

    output = wrapper(images, targets=targets)
    assert output.predictions.decoder.logits.shape == (1, 2, 1, 3)
    assert output.denoising is not None
    assert output.denoising.decoder.logits.shape == (1, 2, 2, 3)
    assert detector_hooks == [None]
    assert generator_hooks == [None]
    (output.predictions.decoder.logits.sum() + output.denoising.decoder.logits.sum()).backward()
    assert detector.head.weight.grad is not None
    assert generator.content.grad is not None
    assert "model.head.weight" in wrapper.state_dict()
    assert "denoising.content" in wrapper.state_dict()
    assert wrapper.checkpoint_config() == {
        "denoising": {
            "target": f"{ToyDenoisingGenerator.__module__}.{ToyDenoisingGenerator.__qualname__}",
            "config": {"metadata_queries": 2},
        },
        "model": {
            "target": f"{ToyDetector.__module__}.{ToyDetector.__qualname__}",
            "config": {"variant": "toy"},
        },
    }


def test_denoising_batch_is_exported_from_training() -> None:
    """Public training import paths resolve to the canonical training contract."""
    assert PublicDenoisingBatch is DenoisingBatch
    assert ModelsDenoisingBatch is DenoisingBatch


def test_denoising_wrapper_rejects_interface_metadata_mismatches() -> None:
    """Composition validates declared dimensions and generated query metadata."""
    wrong_dimensions = ToyDenoisingGenerator()
    wrong_dimensions.num_classes = 4
    with pytest.raises(ValueError, match="classes"):
        DenoisingTrainingModel(ToyDetector(), wrong_dimensions)

    wrapper = DenoisingTrainingModel(ToyDetector(), ToyDenoisingGenerator(metadata_queries=1))
    target = DetectionTarget(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
    with pytest.raises(ValueError, match=r"metadata|quer"):
        wrapper(torch.ones(1, 3, 16, 16), targets=[target])
