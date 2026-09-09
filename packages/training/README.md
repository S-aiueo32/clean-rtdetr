# Clean RT-DETR Training

[日本語版](README.ja.md)

`clean-rtdetr-training` provides single-process matching and losses, contrastive
denoising queries, and an optional Lightning adapter for `clean-rtdetr`. It
requires Python 3.13 or newer.

`SetCriterion` owns the standard VFL/L1/GIoU losses; the former caller-owned
`ClassificationLoss` hook was removed. `HungarianMatcher` offers an explicit
`prediction_box_range="unbounded"` policy for distribution regression. The
Lightning adapter preserves declared `required_callback_keys`
across resume. See [recipe responsibilities](../../docs/recipe-boundaries.md)
for these contracts and the algorithms, data and state owned by the caller.

```sh
python -m pip install clean-rtdetr-training
```

The validated scope is CPU FP32/FP64 loss and gradient parity, plus a small
single-process CPU FP32 Lightning integration with epoch-boundary resume. It
does not claim full-dataset training, convergence, COCO AP, GPU/DDP/AMP,
training augmentation, or distributed normalization. Synthetic data in the
example recipe exercises program wiring and is not a production data pipeline.

For development from source:

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train --help
uv run pytest packages/training/tests
```

From the repository root, run the two-epoch synthetic Lightning example on CPU
with the external configuration tree:

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training
```

The YAML tree is kept at the repository root in `configs/` and is not bundled
in the training distribution. Installed users must provide their own tree or a
copy of the repository's tree. Positional Hydra overrides can follow the two
config options, for example `trainer.max_epochs=1`.

For a small real-image benchmark, the optional `benchmark` extra provides a
Penn-Fudan loader with a reproducible 120/50 train/validation split. See the
[dataset setup and training example](benchmarks/penn_fudan.md).

`PennFudanDetectionDataset` takes `root`, `split`, `seed`, and `val_size`; it
uses the fixed foreground class `person=0`. Dataset factories are constructed
without a model object. Synthetic image size and class count are explicit
configuration values.

For loss construction, the public API accepts normalized cxcywh targets with
zero-based contiguous class labels:

```python
import torch

from clean_rtdetr import DetectionOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion

model_output = DetectionOutput(
    logits=torch.zeros(1, 2, 3, requires_grad=True),
    boxes=torch.full((1, 2, 4), 0.5, requires_grad=True),
)
labels = torch.tensor([1], dtype=torch.int64)
boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
criterion = SetCriterion(matcher=HungarianMatcher())
losses = criterion.stage_losses(model_output, [DetectionTarget(labels=labels, boxes=boxes)])
total_loss = sum(losses.values())
total_loss.backward()
```

For complete training predictions, use `criterion(training_output, targets)`.
The module input is `TrainingOutput`; wrap a core `ModelOutput` in
`TrainingOutput(model_output)` when training without denoising.

The Python API is exposed from `clean_rtdetr_training`; it depends on the
matching `clean-rtdetr` release and does not install the export package. It
does not provide a `clean_rtdetr.training` compatibility shim.

Official training-weight loading is a checkout-only operation through
`tools.compatibility.training.load_official_training_weights`. After loading an
official envelope into a training wrapper, save
`training_model.state_dict()` with `torch.save`; packaged workflows load that
native state dict with `clean_rtdetr.checkpoint.load_model_weights`; its
SHA-256 can be checked optionally. Official weights are initialization, not Lightning resume
checkpoints.

Clean RT-DETR is Apache-2.0 licensed and is based on `lyuwenyu/RT-DETR` at
commit `068dfde65f2667ad6555883c69d73de886518cad`; the workspace root retains
the upstream source, license, and detailed provenance.
