# Matching and training losses

`clean_rtdetr.training` implements the pinned R50vd criterion for single-process
FP32/FP64 PyTorch predictions: Hungarian assignment, Varifocal classification,
normalized cxcywh L1 loss, and GIoU loss. Denoising construction/losses, training
augmentation, optimizer recipes, training/resume, distributed normalization,
and mixed precision remain pending. No convergence or trained accuracy is claimed.

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
from clean_rtdetr.training import DetectionTarget, SetCriterion

# model is an RTDETR instance; images and targets share its device and dtype.
model.train()
output = model(images)  # Default DecoderPlan retains all decoder layers.
targets = [DetectionTarget(labels=labels, boxes=boxes)]  # One target per image.
losses = SetCriterion()(output, targets)
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
intermediate predictions cannot be supervised. Inputs must contain only matching
queries, with no prepended denoising queries.

| Term | Matcher cost weight | Returned loss weight |
| --- | --- | --- |
| Classification | 2 (focal cost, alpha 0.25, gamma 2) | 1 (Varifocal, alpha 0.75, gamma 2) |
| L1 boxes | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` and `CriterionConfig` expose these weights and focal parameters.
Base keys are `loss_vfl`, `loss_bbox`, and `loss_giou`. Auxiliary keys append
`_aux_0`, `_aux_1`, etc.; the encoder uses the final auxiliary index, matching
upstream ordering. Every term is divided by `max(total targets in batch, 1)`,
including when there are more targets than queries. An all-empty batch retains
negative classification loss and differentiable zero box losses. Varifocal IoU
targets and probability weights are detached, preserving upstream gradients.
Active distributed execution with multiple ranks is rejected because local
target normalization would be incorrect.

## Validation scope

CPU tests compare assignment indices, weighted losses, and prediction gradients
against SHA-256-pinned upstream matcher/criterion methods and box geometry.
They cover empty and mixed batches, rectangular assignment, auxiliary supervision,
and malformed input. A small model backward check covers the connection to the
existing model. These checks do not establish full training reproduction.
