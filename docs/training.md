# Matching and training losses

`clean_rtdetr.training` implements the pinned R50vd criterion for single-process
FP32/FP64 PyTorch predictions: Hungarian assignment, Varifocal classification,
normalized cxcywh L1 loss, GIoU loss, and contrastive denoising construction/losses.
A separate [Lightning adapter](lightning.md) implements the baseline optimizer
settings and single-process CPU FP32 training with epoch-boundary resume.
Hydra config groups inject baseline components and optimizer/scheduler factories.
Training augmentation, distributed normalization, mixed precision, and research-method
composition remain pending. No convergence or trained accuracy is claimed.

## Installation and CPU example

```sh
uv sync --locked --extra training
uv run python examples/training_losses.py
```

SciPy supplies `scipy.optimize.linear_sum_assignment`, the assignment solver used
upstream. It is imported only when matching nonempty targets; inference and
package import do not require it. PyTorch computes losses and gradients.
The example runs a synthetic CPU backward pass without datasets or checkpoints.

## API and targets

```python
from clean_rtdetr.training import DetectionTarget, HungarianMatcher, SetCriterion

# model is an RTDETR instance; images and targets share its device and dtype.
model.train()
output = model(images)  # Default DecoderPlan retains all decoder layers.
targets = [DetectionTarget(labels=labels, boxes=boxes)]  # One target per image.
losses = SetCriterion(matcher=HungarianMatcher())(output, targets)
total_loss = sum(losses.values())  # Values are already weighted.
total_loss.backward()
```

Each target contains int64 labels `[N]` and floating-point boxes `[N,4]`.
Labels are contiguous zero-based class indices in `[0,C)`, **not COCO category
IDs**; there is no explicit background class. Boxes use normalized
`(center_x, center_y, width, height)` components in `[0,1]`, with positive target
widths and heights. Zero-area predicted boxes are accepted. Corner coordinates
may extend outside the image; boxes are not clipped by the criterion.
Labels, boxes, and predictions must share a device; target boxes must also
share the prediction dtype. Predictions and targets must be finite.
Empty images use labels of shape `[0]` and boxes of shape `[0,4]`.
Ignore/crowd handling and coordinate conversion belong to the training input pipeline.

`HungarianMatcher()(detection_output, targets)` returns one `MatchIndices`
per image, with int64 `prediction_indices` and `target_indices` on the prediction
device. Assignment is nondifferentiable. Rectangular assignment matches
`min(number_of_queries, number_of_targets)` pairs.

`SetCriterion` accepts a `DetectionOutput` for one stage or a `ModelOutput`
for final and auxiliary supervision. For `ModelOutput`, retained intermediate
decoder predictions are followed by the encoder auxiliary prediction. Every
stage is matched independently. Use the default
`DecoderPlan(collect_intermediate=True)` for all decoder losses; discarded
intermediate predictions cannot be supervised. The main decoder field contains
only matching queries; the optional `denoising` field carries DN predictions and
their known target assignments separately.

| Term | Matcher cost weight | Returned loss weight |
| --- | --- | --- |
| Classification | 2 (focal cost, alpha 0.25, gamma 2) | 1 (Varifocal, alpha 0.75, gamma 2) |
| L1 boxes | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` and `CriterionConfig` expose these weights and focal parameters.
`SetCriterion` requires the abstract `Matcher` module interface for assignment;
the caller or Hydra configuration constructs the matcher. Custom matchers implement
`forward` and `checkpoint_config` without needing Hungarian-specific configuration.
Base keys are `loss_vfl`, `loss_bbox`, and `loss_giou`. Auxiliary keys append
`_aux_0`, `_aux_1`, etc.; the encoder uses the final auxiliary index, matching
upstream ordering. Every term is divided by `max(total targets in batch, 1)`,
including when there are more targets than queries. An all-empty batch retains
negative classification loss and differentiable zero box losses. Varifocal IoU
targets and probability weights are detached, preserving upstream gradients.
Active distributed execution with multiple ranks is rejected because local
target normalization would be incorrect.

## Contrastive denoising

```python
from hydra.utils import instantiate
from clean_rtdetr.configuration import compose_model_config
from clean_rtdetr.training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    HungarianMatcher,
    SetCriterion,
)

training_model = DenoisingTrainingModel(
    instantiate(compose_model_config("r50vd"), _convert_="all"),
    DenoisingQueryGenerator(80, 256, DenoisingConfig()),
).train()
output = training_model(images, targets=targets)
losses = SetCriterion(matcher=HungarianMatcher())(output, targets)
sum(losses.values()).backward()
```

The wrapper owns the detector and learnable `denoising.class_embed` together.
It implements `TrainingModel[ModelOutput]`; `SetCriterion` implements
`TrainingCriterion[DetectionOutput | ModelOutput]`. These optional integration
interfaces are described in [Lightning training](lightning.md#model-and-criterion-interfaces).
Use `training_model.parameters()` for optimizer construction and its `state_dict()`
for weights. The core detector receives no dataset targets. In evaluation mode,
call the wrapper without targets; it delegates to the inference model and does
not draw DN noise. `output.final` always contains only matching queries.

Defaults follow upstream: `num_denoising=100`, `label_noise_ratio=0.5`, and
`box_noise_scale=1.0`. For the largest target count `M` in a batch, the builder
uses `G=max(1, num_denoising // M)` groups and `D=2*M*G` queries, including
positive/negative slots and padding. The requested count is not a hard cap.
Matching queries cannot attend to DN slots; DN groups cannot attend to each
other. Label replacement occurs with probability `label_noise_ratio / 2`.
Box perturbation, clipping, RNG ordering, and the initial reference detach follow
the pinned implementation. In particular, upstream skips inverse sigmoid when
`box_noise_scale=0`; this edge-case behavior is preserved for parity.

Calling a generator as `generator(targets, num_queries=Q)` returns a `DenoisingBatch` of
prepared queries and metadata. `num_denoising=0` or an all-empty target batch
returns `None`, without drawing noise; the wrapper then uses the ordinary path.
A nonempty batch is required, even when DN is disabled. Targets and images must
share the embedding's FP32/FP64 dtype and device. The existing target validation
also applies before disabled/empty handling.

DN loss keys are `loss_vfl_dn_i`, `loss_bbox_dn_i`, and `loss_giou_dn_i` for each
retained layer, including the final layer. They use known positive target slots
and normalize by `max(total targets, 1) * G`; they do not run Hungarian matching.
Negative and padding slots receive negative classification supervision. Use the
default decoder plan to supervise every layer. No DN keys are returned when
construction returns `None`.

`load_official_model_weights(..., model=training_model, ...)` validates and loads
the official DN embedding along with the detector before mutating any parameters.
The inference model loader continues to exclude that embedding. This loads
weights only; use the [Lightning adapter](lightning.md) for resumable training.

## Validation scope

CPU tests compare assignment indices, weighted losses, and prediction gradients
against SHA-256-pinned upstream matcher/criterion methods and box geometry.
They cover empty and mixed batches, rectangular assignment, auxiliary supervision,
and malformed input. A small model backward check covers the connection to the
existing model. These checks do not establish full training reproduction.
DN tests additionally compare seeded construction, attention masks, metadata,
embedding gradients, and weighted per-layer loss/prediction gradients against
pinned upstream code. Wrapper tests cover registration, empty batches, query
separation, inference behavior, and a synthetic backward pass.
