# Clean RT-DETR Training

[日本語版](README.ja.md)

`clean-rtdetr-training` provides single-process matching and losses, contrastive
denoising queries, and an optional Lightning adapter for `clean-rtdetr`. It
requires Python 3.13 or newer.

```sh
python -m pip install clean-rtdetr-training
```

The validated scope is CPU FP32/FP64 loss and gradient parity, plus a small
single-process CPU FP32 Lightning integration with epoch-boundary resume. It
does not claim full-dataset training, convergence, COCO AP, GPU/DDP/AMP,
training augmentation, or distributed normalization. Synthetic data in the
default recipe exercises program wiring and is not a production data pipeline.

For development from source:

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train --help
uv run pytest packages/training/tests
```

Run the packaged two-epoch synthetic Lightning example on CPU with:

```sh
uv run clean-rtdetr-train
```

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
losses = criterion(model_output, [DetectionTarget(labels=labels, boxes=boxes)])
total_loss = sum(losses.values())
total_loss.backward()
```

The Python API is exposed from `clean_rtdetr_training`; it depends on the
matching `clean-rtdetr` release and does not install the export package. It
does not provide a `clean_rtdetr.training` compatibility shim.

Clean RT-DETR is Apache-2.0 licensed and is based on `lyuwenyu/RT-DETR` at
commit `068dfde65f2667ad6555883c69d73de886518cad`; the workspace root retains
the upstream source, license, and detailed provenance.
