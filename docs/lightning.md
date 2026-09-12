# Lightning training and epoch-boundary resume

English | [日本語](ja/lightning.md)

`clean_rtdetr_training.lightning` adapts an injected `TrainingModel` and
`TrainingCriterion` to Lightning automatic optimization. It supplies the
RT-DETR training step, optimizer parameter grouping, EMA, and application RNG
state; `Trainer` owns optimization, clipping, scheduling, logging, and standard
checkpoint serialization. Use the shipped recipe to verify composition and
epoch-boundary restart, then provide your own dataset and recipe for real work.

The validated profile is single-process CPU `32-true`/`bf16-mixed` and CUDA
`32-true`/`16-mixed`, with one DataLoader, `num_workers=0`, and
`accumulate_grad_batches=1`. Lightning rejects other precision modes,
multi-rank execution, gradient accumulation, and worker-based exact resume.
The native criterion has separate DDP loss normalization; see
[GPU execution](gpu-validation.md). COCO augmentation, convergence, and
arbitrary mid-epoch resume are outside this adapter's scope.

<a id="install-and-run"></a>

## Run the supplied smoke recipe

From the repository root:

```sh
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.max_epochs=1 output_dir=outputs/lightning
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.max_epochs=2 output_dir=outputs/lightning \
  resume=outputs/lightning/last.ckpt
```

`trainer.max_epochs` is the total target epoch count, including restored epochs.
The default recipe uses a compact, partly frozen R50vd model and synthetic
128×128 images. It downloads neither weights nor data and demonstrates execution
and restart, not an official model or accuracy. CPU BF16 uses autocast while
model and optimizer state remain FP32:

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.precision=bf16-mixed trainer.max_epochs=1 output_dir=outputs/lightning-bf16
```

Select the shipped CUDA Trainer profile with `trainer=cuda`; it uses
`16-mixed` and `deterministic: warn` because CUDA `grid_sample` backward is not
deterministic.

<a id="hierarchical-configuration-and-dependency-injection"></a>
<a id="application-composition"></a>

## Configuration

The training wheel does not bundle YAML. The CLI requires an explicit external
Hydra tree (`--config-dir` and `--config-name`), then accepts positional Hydra
overrides. `compose_training_config(name, overrides=(), *, config_dir=...)`
provides the same composition API; relative config paths resolve from the current
directory.

The root recipe selects `module`, `datamodule`, and `trainer`. The module recipe
selects the training model, criterion, matcher, parameter-group policy,
optimizer factory, and scheduler factory. Use a group override to replace a
component and a dotted override to set one value:

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  module/optimizer_factory=sgd module/scheduler_factory=step \
  trainer.max_epochs=1 output_dir=outputs/sgd
```

The default model is a compact wiring example. Named factory presets can be
selected with `module/training_model/model=rtdetrv2_r18vd`; set their class count
with `module.training_model.model.num_classes=80`. Presets initialize random
weights. The synthetic data factory has an explicit image size and class count;
it does not inspect the model. Keep the selected image size positive and
stride-aligned, and ensure projected memory has at least the configured query
count.

Optimizer and scheduler settings are Hydra partial factories. At
`configure_optimizers`, the adapter binds actual parameters and verifies every
trainable parameter is owned exactly once, including DN and trainable criterion
parameters. The default recipe uses AdamW (`lr=1e-4`, backbone `1e-5`, betas
`(0.9, 0.999)`, weight decay `1e-4`), an epoch `MultiStepLR`, and norm clipping
at `0.1`. These settings are a baseline wiring configuration, not a full
training-reproduction claim.

<a id="model-and-criterion-interfaces"></a>

## Model and criterion contracts

The adapter depends on interfaces rather than a fixed model layout:

| Interface | Required operations |
| --- | --- |
| `TrainingModel[PredictionT]` | `forward(images, targets=...)`, `predict(images)`, `checkpoint_config()` |
| `TrainingCriterion[PredictionT]` | `forward(output, targets)`, `checkpoint_config()` |
| `Matcher` | `forward(predictions, targets)`, `checkpoint_config()` |
| `DenoisingGenerator` | `forward(targets, num_queries=...)`, dimensions/device/dtype, `checkpoint_config()` |

`forward` returns the payload consumed by the paired criterion; `predict` is the
target-free path and preserves train/eval mode. A criterion returns weighted,
differentiable scalar terms whose sum is optimized. The baseline
`DenoisingTrainingModel` and `SetCriterion` use `TrainingOutput`, whose `.final`
contains matching predictions only. Each batch is FP32 RGB images `[B, 3, H, W]`
and one `DetectionTarget` per image; see [target contracts](training.md#api-and-targets).

`checkpoint_config()` must describe every non-tensor setting that affects a
continuation, using primitives, sequences, and mappings. Tensor state belongs in
the normal PyTorch state dict. Custom injected modules that affect behavior must
implement it; opaque modules can run normally but fail when a resume recipe is
requested. The checkpoint schema is version 2; earlier unreleased adapter
checkpoints are intentionally rejected.

<a id="state-and-resume"></a>
<a id="state-and-reproducibility"></a>

## State and resume

Lightning saves model, optimizer, scheduler, epoch, global step, loop state, and
callback state. The supplied callbacks add:

- `RTDETREMA`: a frozen copy of `training_model`, its update count, and decay
  settings. It updates only after a successful optimizer step, so AMP overflow
  skips do not advance EMA.
- `EpochBoundaryState`: Python, NumPy, PyTorch CPU, active CUDA-device RNG, and
  the shared DataLoader-generator state. It rejects a checkpoint made during an
  incomplete epoch.
- Recipe and execution metadata: component configuration, trainable parameter
  names, precision, device type, clipping, and accumulation. A resume with a
  different setting is rejected; required callbacks must also be present.

Exact epoch-boundary restart also requires the same dataset, batching, sampler,
transforms, environment, and one DataLoader using the supplied CPU generator.
Those data-side details are not serialized. Stateful/streaming datasets, worker
RNGs, a changed world size, and arbitrary mid-epoch restart are not covered.
Loading model weights alone initializes a run; it does not restore optimizer
history.

After fitting, `module.training_model.model` is the live detector and
`ema.module` is the separate evaluation-mode wrapper. To use the detector with
prediction or export CLIs, save `training_model.model.state_dict()`; the official
inference converter does not produce a Lightning resume checkpoint.

<a id="validation"></a>

## Validation scope

Acceptance tests compare uninterrupted training with a fresh module and Trainer
resumed from a completed epoch. They cover shuffled synthetic data, stochastic
inputs, DN noise, empty targets, an LR milestone, and CPU FP32/BF16 plus CUDA
FP32/AMP profiles. They compare live weights, optimizer and scheduler state,
EMA, steps, and RNG; CUDA continuation uses documented tolerances. This is not a
full-dataset or multi-rank Lightning validation.
