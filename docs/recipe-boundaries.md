# Responsibilities learned from the recipes

[日本語版](ja/recipe-boundaries.md)

Clean RT-DETR is a readable PyTorch reference implementation. The four recipes
exercise its extension points; they do not turn the core into a training framework.
Use this guide to decide whether code belongs in a package or in an application,
and to migrate a recipe to the current training contracts.

| Owner | Responsibility | Examples |
| --- | --- | --- |
| `clean_rtdetr` | Model tensor computation, dimensions and observable stage outputs | Features, query assembly, attention masks, decoder predictions, AIFI parameters |
| `clean_rtdetr_training` | Reference RT-DETR supervision and its validated adapter | Targets, matching, DN assignments, VFL/L1/GIoU, distributed target-count normalization, epoch-boundary resume |
| Recipe / application | Research method and experiment composition | MAL, FDR, SAP/O2M, DSI/GAM, data configuration, teacher weights, augmentation and schedules |

Shared recipe code is a candidate for review, not automatically a public API.
An API needs a stable package contract, independence from recipe identity, and
validation independent of that recipe. Method-specific flags, inactive settings,
wrapper introspection, and experimental special cases belong in the recipe.

## Core queries and training assignments

Core `DenoisingQueries` contains prepared content, reference logits and an
attention mask. `QueryAssembler` concatenates these tensors, validates their
layout and owns the initial reference detach boundary. These model operations
do not need target indices, DN groups or loss assignments.

`RTDETR.forward(denoising=...)` receives prepared queries and returns every
decoded slot, with DN slots first. `ModelOutput` has `decoder`, `encoder_aux`,
and `after_aifi`; `final` includes every supplied slot. Core does not assign
supervision to slots.

The training package owns `DenoisingMetadata`, `DenoisingBatch`,
`DenoisingOutput`, `TrainingOutput` and `split_denoising_predictions`.
`DenoisingTrainingModel` passes only `prepared.queries` to the detector.
Its `TrainingOutput.predictions` is a matching-only `ModelOutput`;
`denoising` holds DN predictions and assignments, and `final` is matching-only.
Pass the complete output to the criterion to include DN losses:

```python
output = training_model(images, targets=targets)
losses = criterion(output, targets)
matching_predictions = output.final
aifi_features = output.predictions.after_aifi
```

The public training split helper preserves all collected layers and gradients.
With DN it handles standard `DecoderOutput` only; recipes with extra decoder
tensors own their split. Without DN it returns the original object, preserving
custom payloads. RT-DETRv3 reuses this standard split for each branch while
owning branch masks, target repetition and aggregation.

## Loss reuse without an objective framework

`SetCriterion` implements the reference VFL/L1/GIoU objective. Its
`CriterionConfig` contains exactly those weights and Varifocal parameters.
Another classification equation belongs to another criterion. The former
`ClassificationLoss` hook and its optional `SetCriterion` arguments are
removed: MAL must not leave inactive VFL settings in a checkpoint recipe.

`SetCriterion.forward` consumes `TrainingOutput`, matching its training model's
output contract. Wrap a raw `ModelOutput` with `TrainingOutput` explicitly;
use `criterion.stage_losses(detection_output, targets)` for a single stage.
The public `match_stage` helper provides the corresponding single-stage
assignment and normalizer without widening the training criterion's input type.

The public `clean_rtdetr_training.losses` functions expose reusable operations:

| Function | Contract |
| --- | --- |
| `iter_matched_stages` | Independent final/auxiliary matching and canonical DN assignments, with distributed target-count normalizers |
| `gather_matched_pairs` | Aligned predicted/target boxes and labels, retaining prediction gradients |
| `matched_iou_targets` | Detached IoU scores and a separate mask including zero-IoU positive matches |
| `matched_box_losses` | Unweighted L1/GIoU scalars using the supplied positive normalizer |

These helpers encode standard RT-DETR supervision. They do not choose a
classification equation, own weights, or dispatch recipe types. DEIM composes
them in `MALCriterion`; different assignments, stage aggregation, and
distributed normalization belong there. D-FINE's union matching and
distribution losses remain in its recipe.

`HungarianMatcher(prediction_box_range="unbounded")` stays in training because
normalized coordinates need not be clipped predictions. The default `"unit"`
still requires predictions in `[0, 1]`. The explicit alternative accepts finite
cxcywh predictions outside this interval and validates nonnegative width/height;
target boxes remain normalized and validated. D-FINE selects this contract
without duplicating the cost and assignment solver. This does not add
distribution regression to core or extend export support.

## Application-owned data composition

Datasets and `DetectionDataModule` do not inspect a training model. The CLI
injects the shared DataLoader generator; the datamodule receives a zero-argument
dataset factory with configuration already bound. `TrainingModel` needs neither
a `num_classes` property nor a `.model` attribute for data construction.

The Penn-Fudan loader owns file/mask decoding and the documented `person=0`
label convention. The recipe chooses a compatible head, split and seed.
The synthetic loader receives explicit `num_classes` and `image_size`:

```yaml
dataset_factory:
  _target_: clean_rtdetr_training.data.SyntheticDetectionDataset
  _partial_: true
  num_classes: 3
  image_size: 128
```

A recipe may interpolate shared configuration into model and dataset without
passing a built model. The caller ensures input-stride alignment and enough
projected memory for its query count. There is no `image_size=auto` or
`from_training_model` constructor. RT-DETRv3's 450-query branch, for example,
exceeds the 336 positions produced by a 128 input.

Augmentation, category remapping, validation metrics, private RNGs, batching and
samplers remain application decisions. Reuse a loader where its contract fits;
do not add model-dependent behavior to accommodate a recipe.

## Observable structure and callback state

`HybridEncoder.aifi_parameters()` exposes parameters of an existing stage.
Together with `after_aifi`, it lets callers observe the model without relying
on internal attribute names. Core does not calculate a gradient ratio or know
a teacher, epoch, optimizer or distillation weight. RT-DETRv4 owns DSI/GAM and
uses this structural API to select the parameters.

The optional Lightning adapter retains `required_callback_keys` as opaque
resume requirements. It checks declared keys and preserves saved requirements
even when the resume configuration omits the declaration. It does not recognize
GAM by name or reconstruct its algorithm. Each stateful recipe callback owns
its state and configuration checks; the recipe declares its key to prevent
silently resuming without that callback.

## What stays in each recipe

| Recipe | User-owned behavior |
| --- | --- |
| DEIM | MAL equation, objective configuration and experiment schedule |
| D-FINE | FDR decoder/output, bins, LQE, FGL, union assignments and DDF |
| RT-DETRv3 | SAP masks/selectors, O2M target repetition, CNN auxiliary head and branch weights |
| RT-DETRv4 | Teacher loading/preprocessing, DSI projection/alignment, GAM observation/update and callback state |

All four remain independent checkout projects. Custom outputs and checkpoints
are not automatically accepted by prediction/export CLIs. Official inference
weight conversion remains the private implementation of the packaged core
`clean-rtdetr-convert` CLI. Official training compatibility and stage-parity
helpers remain checkout-only in `tools/compatibility`.

## Source migration and validation scope

- Import DN metadata, batches, outputs and the split helper from
  `clean_rtdetr_training`; prepared `DenoisingQueries` stays in core.
- Read decoder/encoder/AIFI fields through `training_output.predictions`.
  Pass the complete `TrainingOutput` to the criterion for DN supervision.
- Give core `RTDETR` prepared query tensors; its output includes all slots.
- Construct datasets/datamodules with explicit configuration and factories.
  Remove wrapper metadata properties added solely for data construction.
- Compose alternate objectives with `TrainingCriterion`. There is no shim
  for the former `ClassificationLoss` injection API.

Model/DN parameter state-dict keys and the baseline
`SetCriterion.checkpoint_config()` schema are preserved. Source imports,
output types and constructor signatures change. Changed custom criteria or
data configurations do not imply compatible continuation; use the same
complete configuration and validated epoch-boundary resume contract.

Tests cover dependency direction, raw query outputs and gradients, DN splitting,
reference loss parity, model-independent CLI data construction, recipe objectives
and callback resume behavior. These small CPU checks do not establish full
recipe training, convergence, COCO AP, recipe-specific GPU/DDP/AMP execution,
or custom recipe export. Baseline GPU evidence has its own
[validation scope](gpu-validation.md). See [training](training.md),
[Lightning](lightning.md) and the recipe READMEs for operational details.
