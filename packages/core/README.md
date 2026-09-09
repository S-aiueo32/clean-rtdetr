# Clean RT-DETR Core

[日本語版](README.ja.md)

`clean-rtdetr` provides readable PyTorch constructors for the official RT-DETR
and RT-DETRv2 PResNet presets, tensor contracts, image preprocessing, top-k
postprocessing, native checkpoint loading, and COCO prediction/evaluation
adapters. It requires Python 3.13 or newer.

The package installation commands are:

```sh
python -m pip install clean-rtdetr
python -m pip install "clean-rtdetr[inference]"   # Pillow image I/O
python -m pip install "clean-rtdetr[evaluation]"  # pycocotools metrics
```

The base install includes the official R50vd checkpoint converter and can be
used without Pillow, training, or export dependencies:

```sh
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

For development from source, clone the repository and install the workspace
environment:

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --all-packages
uv run pytest packages/core/tests
```

The core package covers image-to-prediction R50vd inference in CPU/PyTorch,
including Pillow/uint8 preprocessing and COCO-format prediction JSON. Use the
`inference` extra for image files. The packaged
`clean_rtdetr.checkpoint.load_model_weights` loader accepts a local native plain
`state_dict` and optionally checks its SHA-256; it does not download weights.
`clean-rtdetr-convert` defaults to the pinned official R50vd EMA source and
hash; `--source` and `--expected-sha256` override those checks.
Fixed-shape ONNX export belongs to `clean-rtdetr-export`, and matching, losses,
denoising, and the Lightning adapter belong to `clean-rtdetr-training`.
The converter is bundled as an internal core CLI implementation and exposes no
public compatibility API. Full, stage, and training compatibility helpers remain
checkout-only; see the [inference guide](../../docs/inference.md).
Full-dataset training, GPU/DDP/AMP validation, COCO AP measurement, and latency
benchmarks are outside the validated scope.

Use the named factory to construct any of the ten upstream PResNet presets with
randomly initialized weights. `get_model_preset()` exposes the matching upstream
YAML and checkpoint URLs as provenance; model construction never downloads them.

```python
from clean_rtdetr import create_model, list_models

print(list_models())
model = create_model("rtdetrv2_r18vd").eval()
```

Architecture construction is covered for the listed presets. Official checkpoint
conversion, full-model parity, image inference, and ONNX parity remain validated
only for `rtdetr_r50vd`.

With the checkout development environment, a model can be instantiated and run on
preprocessed tensors as follows (the example uses random weights):

```python
import torch

from clean_rtdetr import DecoderPlan, rtdetr_r50vd

model = rtdetr_r50vd().eval()
with torch.inference_mode():
    output = model(torch.rand(1, 3, 640, 640), plan=DecoderPlan(collect_intermediate=False))
print(output.final.logits.shape, output.final.boxes.shape)  # [1, 300, 80], [1, 300, 4]
```

The package is Apache-2.0 licensed and is based on pinned revisions of
`lyuwenyu/RT-DETR`. The source repository retains the upstream source, license,
detailed provenance, and parity tests under `third_party/RT-DETR`.
