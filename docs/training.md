# Matching and training losses

English | [日本語](ja/training.md)

`clean_rtdetr_training` provides the pinned R50vd training criterion: Hungarian
matching, Varifocal classification, normalized cxcywh L1 and GIoU losses, and
contrastive denoising (DN) queries and losses. Use it when composing a PyTorch
training loop or a recipe-specific criterion. It validates FP32/FP64 predictions
and targets; image augmentation and training-data conversion are caller-owned.

The criterion supports native PyTorch distributed execution. It sums target
counts across ranks, divides by the world size, then clamps the result to one
before loss normalization, which agrees with DDP gradient averaging. Every rank
must execute the same criterion calls. This does not make the separate
[Lightning adapter](lightning.md) multi-rank capable. No convergence or trained
accuracy result is claimed.

<a id="installation-and-cpu-example"></a>

## Install and smoke test

From the repository root with Python 3.13 or 3.14 and uv:

```sh
uv sync --package clean-rtdetr-training
uv run python tools/training_losses.py
```

The command runs a synthetic CPU backward pass; it needs neither a dataset nor a
checkpoint. SciPy supplies `scipy.optimize.linear_sum_assignment` when matching
nonempty targets. Importing the package and inference do not import it.

<a id="api-and-targets"></a>

## Inputs, outputs, and losses

The public composition types are exported from `clean_rtdetr_training`.
`clean_rtdetr_training.data`, `.models`, and `.lightning` export their own
public types for direct or Hydra construction.

```python
import torch

from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion, TrainingOutput

model = rtdetr_r50vd().train()
images = torch.rand(1, 3, 128, 128)
labels = torch.tensor([0])
boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
output = model(images)  # The default DecoderPlan retains all decoder layers.
targets = [DetectionTarget(labels=labels, boxes=boxes)]
losses = SetCriterion(matcher=HungarianMatcher())(TrainingOutput(output), targets)
sum(losses.values()).backward()  # Returned terms are already weighted.
```

Provide one `DetectionTarget` per image. `labels` is int64 `[N]`, with contiguous
zero-based IDs in `[0, C)`; these are not COCO category IDs and there is no
background class. `boxes` is FP32 or FP64 `[N, 4]`, normalized
`(center_x, center_y, width, height)` values in `[0, 1]` with positive width and
height. Targets and predictions must be finite and share device; target boxes
must also share prediction dtype. An empty image uses `[0]` labels and `[0, 4]`
boxes. Predicted boxes may have zero area; corners outside the image are not
clipped. Convert coordinates and handle crowd/ignore annotations before this API.

`HungarianMatcher()(detection_output, targets)` returns one `MatchIndices` per
image. Its int64 prediction and target indices are on the prediction device;
assignment is nondifferentiable and matches at most `min(queries, targets)`
pairs.

`SetCriterion` accepts `TrainingOutput`, which contains final, auxiliary, and
optional DN supervision. Wrap a core `ModelOutput` as `TrainingOutput(output)`
when DN is absent. For one `DetectionOutput`, use
`criterion.stage_losses(output, targets)`.

| Term | Matcher cost | Returned loss weight |
| --- | --- | --- |
| Classification | 2; focal, alpha 0.25, gamma 2 | 1; Varifocal, alpha 0.75, gamma 2 |
| L1 box | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` and `CriterionConfig` expose those weights and focal parameters.
Base keys are `loss_vfl`, `loss_bbox`, and `loss_giou`; decoder auxiliary keys
append `_aux_0`, `_aux_1`, and so on, with the encoder at the final auxiliary
index. Each term uses the global average target count described above. An
all-empty batch retains negative classification loss and differentiable zero box
losses. Varifocal IoU targets and probability weights are detached.

The standard criterion owns only VFL, L1, and GIoU. A recipe that changes the
classification equation, matching policy, or DN supervision owns that policy;
it may reuse `iter_matched_stages`, `gather_matched_pairs`,
`matched_iou_targets`, and `matched_box_losses` from
`clean_rtdetr_training.losses`. Custom matchers implement `forward` and
`checkpoint_config`.

<a id="contrastive-denoising"></a>

## Contrastive denoising

```python
from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr_training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    HungarianMatcher,
    SetCriterion,
)

training_model = DenoisingTrainingModel(
    rtdetr_r50vd(), DenoisingQueryGenerator(80, 256, DenoisingConfig())
).train()
# Reuse images and targets from the preceding example.
training_output = training_model(images, targets=targets)
losses = SetCriterion(matcher=HungarianMatcher())(training_output, targets)
sum(losses.values()).backward()
```

`DenoisingTrainingModel` owns both the detector and learnable
`denoising.class_embed`; use its `parameters()` and `state_dict()`. The core
detector receives prepared query tensors, never raw dataset targets. The wrapper
splits the DN prefix: `TrainingOutput.predictions` and `.final` contain matching
queries, while `.denoising` carries DN predictions and metadata. Pass the whole
output to train matching and DN losses, or use
`TrainingOutput(training_output.predictions)` to omit DN supervision. In eval
mode call the wrapper without targets; it delegates to target-free inference.

Defaults are `num_denoising=100`, `label_noise_ratio=0.5`, and
`box_noise_scale=1.0`. If the largest target count is `M`, DN creates
`G = max(1, num_denoising // M)` groups and `2 * M * G` slots, so the requested
count is not a cap. `num_denoising=0` or an all-empty batch returns no DN batch.
DN loss keys are `loss_vfl_dn_i`, `loss_bbox_dn_i`, and `loss_giou_dn_i` for
each retained decoder layer; they use known positive assignments rather than
Hungarian matching and normalize by `global_average_target_count * G`, where
the global average is clamped to one. This is `max(total targets, 1) * G` in
single-process execution.

For compatibility loading, `tools.compatibility.training.load_official_training_weights`
is checkout-only and initializes a complete wrapper from an official envelope.
For packaged use, save the initialized wrapper with
`torch.save(training_model.state_dict(), path)` and load it into the reconstructed
wrapper with `clean_rtdetr.checkpoint.load_model_weights(path, model=training_model)`.
The inference converter does not create a training
checkpoint. Use [Lightning training](lightning.md#state-and-resume) when the
optimizer, scheduler, EMA, and epoch-boundary resume state are required.

<a id="validation-scope"></a>

## Validation scope

CPU parity tests compare assignment, weighted losses, and prediction gradients
with pinned upstream matcher/criterion behavior. They cover empty and mixed
batches, rectangular assignment, auxiliary supervision, DN construction and
losses, wrapper query separation, and a synthetic backward pass. The two-rank
loss test covers unequal and empty target counts. These checks do not reproduce
full training, augmentation, or trained accuracy.
