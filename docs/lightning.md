# Lightning training and epoch-boundary resume

`clean_rtdetr.training.lightning` provides a thin adapter around the existing
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
uv sync --locked --extra lightning
uv run python examples/lightning_training.py trainer.max_epochs=1 output_dir=outputs/lightning
uv run python examples/lightning_training.py trainer.max_epochs=2 output_dir=outputs/lightning \
  resume=outputs/lightning/last.ckpt
```

`trainer.max_epochs` is the total target epoch count, including restored epochs.
The default recipe uses a compact detector with a partially frozen R50vd backbone
and synthetic 64×64 images. It downloads no checkpoints or datasets. It
exercises execution and restart, not an official trained model or accuracy.

Lightning and Hydra are optional consumer dependencies. Importing `clean_rtdetr`
or `clean_rtdetr.training` does not import them. The development environment
includes them so adapter regression tests and type checks always run.

## Hierarchical configuration and dependency injection

The application uses Hydra's Compose API and `hydra.utils.instantiate`.
Configuration lives under `configs/`; no configured target imports `examples`.

```text
configs/
  lightning_training.yaml
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
feature specs or dimensions. `builtins.tuple` preserves the typed feature-spec
contract while constructing the YAML sequence.

The Lightning module receives the DN wrapper, criterion, parameter-group policy,
optimizer factory, and scheduler factory through constructor injection. Optimizer
and scheduler configs use `_partial_: true`: Hydra constructs factories first,
then `configure_optimizers` binds the actual trainable parameters and optimizer.
The adapter does not instantiate a fixed optimizer or scheduler class.

For example, change both optimizer and scheduler through config groups:

```sh
uv run python examples/lightning_training.py \
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

Dataset, DataLoader, callbacks, and Trainer are also instantiated from config.
The application injects the **same** generator object into the DataLoader and
`EpochBoundaryState`; it does not instantiate two generators from repeated YAML.
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

The live detector is available as `module.training_model.model`. After fitting,
`ema.module` holds the separate EMA wrapper in evaluation mode. These remain
ordinary PyTorch modules, so inference does not require a Lightning Trainer.
The existing official checkpoint loader can initialize `training_model` before
constructing the Lightning adapter; official weights are not Lightning resume
checkpoints.

## Validation

The acceptance test compares uninterrupted training against a fresh module and
Trainer resumed from an epoch checkpoint. It exercises shuffled data, stochastic
inputs, DN noise, empty targets, and an LR milestone, then compares live weights,
optimizer moments, scheduler progress, EMA state, step counts, and RNG state.
The exact restart test uses a fresh interpreter without probabilistic runtime
type-check instrumentation, which itself consumes Python RNG state. Separate
tests compare EMA updates against the pinned upstream implementation.
These checks establish this CPU restart boundary, not training reproduction or AP.

Upstream training defaults and EMA provenance are recorded in
[third_party/RT-DETR](../third_party/RT-DETR/README.md). Lightning's interfaces are
documented in its [LightningModule guide](https://api.lightning.ai/docs/pytorch/stable/common/lightning_module.html)
and [checkpoint guide](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/checkpointing_basic.rst).
