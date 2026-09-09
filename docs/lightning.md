# Lightning training and epoch-boundary resume

English | [日本語](ja/lightning.md)

`clean_rtdetr_training.lightning` provides a thin adapter around the existing
DN training model. Lightning's `Trainer` owns automatic optimization, gradient
clipping, epoch scheduling, logging, and checkpoint serialization/restoration.
This project supplies the RT-DETR loss step, optimizer parameter groups, EMA,
and application RNG callback state.

The validated profile is **single-process CPU FP32**, with one optimizer step
per batch, one DataLoader, a dedicated CPU generator, and `num_workers=0`.
Tests use synthetic data. COCO training augmentation, full training convergence,
GPU/DDP/AMP, arbitrary mid-epoch resume, and research-method composition remain pending.

## Install and run

```sh
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train \
  trainer.max_epochs=1 output_dir=outputs/lightning
uv run clean-rtdetr-train \
  trainer.max_epochs=2 output_dir=outputs/lightning \
  resume=outputs/lightning/last.ckpt
```

`trainer.max_epochs` is the total target epoch count, including restored epochs.
The default recipe uses a compact detector with a partially frozen R50vd backbone
and synthetic 64×64 images. It downloads no checkpoints or datasets. It
exercises execution and restart, not an official trained model or accuracy.

Lightning is owned by the `clean-rtdetr-training` distribution. Installing or
importing the core `clean-rtdetr` package does not install or import the training
stack. Hydra is confined to the training distribution and its experiment recipes.

## Hierarchical configuration and dependency injection

The application uses Hydra's Compose API and `hydra.utils.instantiate` through
`clean_rtdetr_training.compose_training_config`. Configuration is packaged in
the training distribution; no configured target imports a command-line module.

```python
from clean_rtdetr_training import compose_training_config

config = compose_training_config("lightning_training")
```

```text
packages/training/src/clean_rtdetr_training/configs/
  lightning_training.yaml
  criterion/set.yaml
  matcher/hungarian.yaml
  denoising/contrastive.yaml
  datamodule/synthetic.yaml
  model/
    compact.yaml
    backbone/r50vd.yaml
    encoder/compact.yaml
    projector/compact.yaml
    selector/compact.yaml
    decoder/compact.yaml
  optimizer/
    adamw.yaml
    sgd.yaml
  scheduler/
    multistep.yaml
    step.yaml
```

The model group recursively instantiates package components: `PResNet`,
`HybridEncoder`, `MemoryProjector`, `QuerySelector`, `QueryAssembler`, and
`Decoder`, then injects them into `RTDETR`. Shared dimensions are interpolated
from the root configuration. Existing component validation rejects incompatible
feature specs or dimensions. Constructors accept feature specs as a sequence,
so YAML uses ordinary lists; each module stores an immutable tuple internally.

The Lightning module receives a training model, criterion, parameter-group policy,
optimizer factory, and scheduler factory through constructor injection. Optimizer
and scheduler configs use `_partial_: true`: Hydra constructs factories first,
then `configure_optimizers` binds the actual trainable parameters and optimizer.
The adapter does not instantiate a fixed optimizer or scheduler class.

For example, change both optimizer and scheduler through config groups:

```sh
uv run clean-rtdetr-train \
  optimizer@module.optimizer_factory=sgd \
  scheduler@module.scheduler_factory=step \
  trainer.max_epochs=1 output_dir=outputs/sgd
```

The default AdamW recipe follows pinned R50vd settings: LR `1e-4`, backbone LR
`1e-5`, betas `(0.9, 0.999)`, weight decay `1e-4`, and an epoch-based
`MultiStepLR(milestones=[1000], gamma=0.1)`. `RTDETRParameterGroups` is an
independently injectable callable that maps parameter names to upstream names
before applying encoder/decoder bias and `norm.*weight` decay exclusions. All
trainable parameters, including DN embeddings, must belong to the optimizer
exactly once. The Trainer controls clipping; the recipe sets norm limit `0.1`.
Alternative factories demonstrate DI, not upstream training recipe parity.

## Model and criterion interfaces

The adapter depends on `TrainingModel[PredictionT]` and
`TrainingCriterion[PredictionT]`, not on `DenoisingTrainingModel` or `SetCriterion`.
Both are abstract `nn.Module` subclasses so implementations retain normal
parameter/buffer registration, forward hooks, device transfer, and state dicts.
The generic prediction type connects the model to its paired criterion; the
Lightning adapter does not inspect the prediction payload.

| Interface | Required operations |
| --- | --- |
| `TrainingModel[PredictionT]` | `forward(images, targets=...)`, `predict(images)`, `checkpoint_config()` |
| `TrainingCriterion[PredictionT]` | `forward(output, targets)`, `checkpoint_config()` |
| `Matcher` | `forward(predictions, targets)`, `checkpoint_config()` |
| `DenoisingGenerator` | `forward(targets, num_queries=...)`, dimensions/device/dtype, `checkpoint_config()` |
| `Detector` | `forward(images, denoising=..., plan=..., collect_aifi=...)`, dimensions, `checkpoint_config()` |

`SetCriterion` requires an injected implementation of the abstract `Matcher`
interface. The baseline Hydra config explicitly constructs `HungarianMatcher`;
another implementation can be selected through its `_target_`.
Matchers return one `MatchIndices` assignment per image on the prediction device
and validate the detection/target inputs. The criterion obtains resume settings
through `checkpoint_config()`, without requiring a matcher `.config` attribute.

The DN wrapper depends on `Detector` and `DenoisingGenerator`. It uses their
public dimensions, device, dtype, and checkpoint configuration rather than
inspecting decoder/selector configs or a particular embedding layout. Prepared
DN queries contain their assignment metadata; the detector returns separated
matching and DN predictions, preserving the `ModelOutput.final` contract.

`RTDETR` in turn accepts abstract backbone, encoder, projector, selector,
assembler, and decoder stages from `clean_rtdetr.models.interfaces`. Each stage
declares its tensor boundary, compatibility metadata, and checkpoint settings.
Implementations remain registered PyTorch modules. Core exposes the fixed inference
baseline through `rtdetr_r50vd()` without requiring a configuration framework.
Experimental compositions remain in the training distribution's Hydra config groups.

The baseline factory is available after installing the core wheel:

```python
from clean_rtdetr import rtdetr_r50vd

model = rtdetr_r50vd()
```

The training distribution packages its model groups with the training recipe.
Their shared dimensions are controlled by the root recipe and checked against the
DN generator and dataset during composition and construction.

The `criterion`, `matcher`, and `denoising` config groups expose those choices
alongside the existing model-stage, optimizer, and scheduler groups. For
example, after adding
`packages/training/src/clean_rtdetr_training/configs/matcher/experimental.yaml`,
select it with
`matcher@module.criterion.matcher=experimental`. A replacement config supplies
its own `_target_` and constructor arguments. No command-line module is required.

`predict` is the target-free path and preserves the caller's train/eval mode.
The criterion returns a dictionary of weighted, differentiable scalar losses.
`checkpoint_config()` returns non-tensor settings as primitives, sequences, and
mappings; implementation identity and these settings are checked during resume.
Tensor state stays in PyTorch's `state_dict`. The adapter never reaches into a
model's `.model` or `.denoising`, or a criterion's `.matcher` or `.config`.

Injected decoder layers and selector score/box heads also form part of the
checkpoint recipe, including their concrete types and non-tensor configuration.
The built-in `DecoderLayer`, `BoxHead`, and exact `torch.nn.Linear` type provide
these settings. Every custom concrete type, including subclasses of those types,
must explicitly define `checkpoint_config()` and return a dictionary describing
all settings that affect its behavior. An opaque module remains usable for
ordinary forward execution, but requesting its checkpoint recipe raises
`TypeError`. This prevents a restart from silently changing injected dropout or
head behavior while accepting identically shaped weights.

The baseline DN wrapper and set criterion implement these interfaces. Another
model/criterion pair can use its own prediction type and internal structure, then
be selected through Hydra `_target_` configuration. It need not inherit either
baseline implementation. Keep `parameter_groups_factory: null` for the default
all-trainable-parameters grouping or inject an appropriate policy;
`RTDETRParameterGroups` remains explicitly specific to RT-DETR parameter names.
Trainable criterion parameters are also registered and optimized; parameters
shared with the model are included exactly once.

The interface-based checkpoint recipe uses schema version 2. Checkpoints from
the earlier, unreleased concrete adapter recipe are rejected; use a new run for
this schema rather than treating those files as compatible resumes.

## Application composition

The application instantiates a `LightningDataModule` from the `datamodule` config
group and passes it to `Trainer.fit(..., datamodule=...)`. The default
`DetectionDataModule` receives dataset and DataLoader factories through Hydra
partials. It constructs the dataset during `setup("fit")`, then supplies the
loader through `train_dataloader()`. Repeated setup retains the same dataset.
Collation, batching, and shuffling are configured under the loader factory.
Another experiment can replace the entire datamodule with its own
`LightningDataModule` implementation.

The application injects the **same** generator object into the datamodule and
`EpochBoundaryState`. The datamodule passes that generator to its DataLoader,
preserving the existing epoch-boundary RNG restore contract.
The entry point contains only composition and execution, without a model factory.
It uses a small argument parser plus the Compose API because Hydra 1.3's own CLI
help construction is incompatible with the tested Python 3.14 argparse version.
The supported CLI accepts Hydra override expressions, not Hydra's launcher/multirun flags.

Each batch is `(images, targets)`: FP32 RGB images `[B,3,H,W]` and a list of
`DetectionTarget` instances. See [target contracts](training.md#api-and-targets).
The adapter explicitly transfers immutable targets, retains every decoder layer,
sums all weighted matching and DN losses, and logs `train/loss`. Validation and
COCO metrics remain separate from this training adapter.

## State and reproducibility

Lightning saves model, optimizer, scheduler, epoch, global step, loop progress,
and callback state in its ordinary checkpoint format. We add:

- `RTDETREMA`: a separate frozen copy of the DN wrapper, update count, decay,
  and warmup settings. Floating parameters and buffers follow upstream's
  `decay * (1 - exp(-updates / warmups))`; integer buffers stay at their initial
  values. EMA updates after optimizer steps and never replaces the live weights.
- `EpochBoundaryState`: Python, global NumPy, PyTorch CPU, and the supplied
  DataLoader generator states. The callback restores them through Lightning's
  resume lifecycle and rejects checkpointing incomplete epochs.
- Recipe metadata: factory targets and arguments, loss/DN/model configuration, trainable parameter
  names, and update-affecting Trainer settings. Incompatible resume settings
  are rejected. Required EMA/RNG callbacks must also be supplied when resuming.

Resume requires the same dataset, batch size, sampler configuration, transforms,
and software/runtime environment. Those are caller responsibilities; arbitrary
dataset objects and private augmentation generators are not serialized. Worker
RNGs, stateful streaming datasets, and changing world size are outside this
validated profile. Prefer a completed epoch checkpoint with all states; loading
weights alone is initialization, not continuation of optimizer history.

For the baseline wrapper, the live detector is available as
`module.training_model.model`. After fitting,
`ema.module` holds the separate EMA wrapper in evaluation mode. These remain
ordinary PyTorch modules, so inference does not require a Lightning Trainer.
The checkout-only
`tools.compatibility.training.load_official_training_weights` can initialize
`training_model` before constructing the Lightning adapter. For a packaged
workflow, save `training_model.state_dict()` with `torch.save` and use the core
native `clean_rtdetr.checkpoint.load_model_weights` loader. The installed
`clean-rtdetr-convert` command converts the official envelope for inference only;
official weights are not Lightning resume checkpoints.

## Validation

The acceptance test compares uninterrupted training against a fresh module and
Trainer resumed from an epoch checkpoint. It exercises shuffled data, stochastic
inputs, DN noise, empty targets, and an LR milestone, then compares live weights,
optimizer moments, scheduler progress, EMA state, step counts, and RNG state.
The exact restart test uses a fresh interpreter without probabilistic runtime
type-check instrumentation, which itself consumes Python RNG state. Separate
tests compare EMA updates against the pinned upstream implementation.
An independent model/criterion pair with a custom prediction payload also covers
real Trainer updates and checkpoint restore, forward hooks, trainable criterion
state, shared parameters, and changed implementation/configuration rejection.
These checks establish this CPU restart boundary, not training reproduction or AP.

Upstream training defaults and EMA provenance are recorded in
[third_party/RT-DETR](../third_party/RT-DETR/README.md). Lightning's interfaces are
documented in its [LightningModule guide](https://api.lightning.ai/docs/pytorch/stable/common/lightning_module.html)
and [checkpoint guide](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/checkpointing_basic.rst).
