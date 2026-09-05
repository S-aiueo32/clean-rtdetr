# Clean RT-DETR

A readable RT-DETR reference implementation with composable research extensions.
The design separates model variants, training methods, and experiment recipes
from deployment transformations.

## Current implementation

The current implementation connects **images → R50vd backbone → HybridEncoder →
projected memory → selected queries → iterative decoder predictions** in PyTorch.

- Frozen feature/query contracts with jaxtyping shape/dtype annotations and eager validation.
- Decoder input projection, row-major flattening, and explicit level offsets.
- Anchor generation and top-k selection preserving upstream masking semantics.
- Feature-content detach and differentiable encoder auxiliary predictions.
- DN-ready query assembly with an explicit reference-logit detach boundary.
- Self-attention, multi-scale deformable cross-attention, and iterative box refinement.
- Explicit decoder execution/output plans instead of implicit inference-layer selection.
- Forward and gradient parity tests against SHA-256-pinned upstream methods.
- An explicit R50vd factory and component-injected `RTDETR`, with optional AIFI features.
- Strict official checkpoint loading for all inference stages, with a reproducible parity example.
- Shared Pillow/uint8 image preprocessing and upstream-compatible top-k postprocessing.
- Fixed-shape ONNX export through the `FixedShapeRTDETR` wrapper, with CPU FP32
  ONNX Runtime parity verification.
- Single-process Hungarian matching and Varifocal/L1/GIoU losses, with CPU
  upstream loss/gradient parity and decoder/encoder auxiliary supervision.
- Contrastive denoising query construction, a training wrapper with registered
  class embeddings, and per-layer DN losses with CPU upstream parity.
- Optional Lightning training adapter with baseline AdamW groups, EMA, and
  tested CPU FP32 epoch-boundary resume; see [Lightning training](docs/lightning.md).
- Hierarchical Hydra configs for model-stage and training-component injection.

COCO-format prediction JSON and optional official bbox evaluation are available
as separate commands; see [COCO evaluation](docs/evaluation.md). pycocotools is
loaded only by the metric adapter, and is unnecessary for prediction generation.

See [training losses](docs/training.md) for the target contract and a synthetic
CPU backward example. SciPy is an optional training dependency for assignment.

Full dataset training, GPU/DDP/AMP validation, research-method composition, and COCO AP measurement
remain to be implemented. No AP or latency is claimed; export is limited to the fixed
batch-1 640×640 CPU FP32 profile described below.
`examples/query_selection.py` continues to demonstrate the component path using
synthetic features and randomly initialized weights.

## Image-to-prediction API

```python
import torch
from hydra.utils import instantiate
from clean_rtdetr import DecoderPlan
from clean_rtdetr.configuration import compose_model_config

model = instantiate(compose_model_config("r50vd"), _convert_="all").eval()  # Random weights.
with torch.inference_mode():
    output = model(torch.rand(1, 3, 640, 640), plan=DecoderPlan(collect_intermediate=False))
print(output.final.logits.shape, output.final.boxes.shape)  # [1,300,80], [1,300,4]
```

Inputs and outputs are raw model tensors. See [inference I/O](docs/inference.md)
for their contracts, image conversion, official weights, and parity commands.

## Image preprocessing and detections

Install Pillow support with `uv sync --locked --extra inference` (or combine it
with export as `uv sync --locked --extra export --extra inference`), then run the
complete path with local official weights:

```sh
uv run python examples/image_inference.py image.jpg \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45
```

See [inference I/O](docs/inference.md) for the Python API, tensor contracts, and
postprocessing semantics. The example does not download artifacts or establish
COCO AP or latency.

## Fixed-shape ONNX export

Install the export dependencies with `uv sync --locked --extra export`. The
`export_onnx` API exports a `FixedShapeRTDETR` graph with raw logits
`[1,300,80]` and normalized cxcywh boxes `[1,300,4]`:

```python
from pathlib import Path
from hydra.utils import instantiate
from clean_rtdetr.configuration import compose_model_config
from clean_rtdetr.export import export_onnx

model = instantiate(
    compose_model_config("r50vd"), _convert_="all"
).eval()  # Load trained weights separately.
report = export_onnx(model, Path("outputs/rtdetr.onnx"))
```

For a checkpoint-backed export, use the CLI. It requires a local official
checkpoint and its SHA-256; weights are never downloaded:

```sh
uv run python examples/export_onnx.py \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --output outputs/rtdetr.onnx \
  --report outputs/rtdetr.json
```

The exporter uses fixed batch 1 and 640×640 CPU FP32, ONNX opset 18, and
compares eager and ONNX Runtime outputs for seeds 41 and 42 (`atol=1e-5`,
`rtol=1e-4`). Both exporter and Runtime graph optimizations are disabled in
the validated profile; see [runtime settings](docs/inference.md#fixed-shape-onnx-export).
The [official-weight report](benchmarks/onnx-parity.json) records errors and
reproducibility metadata. Dynamic shapes, TensorRT, AP, and latency are not supported.

## Contributing

Development setup, validation commands, linting policy, and implementation notes
are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

For a quick local check, install the locked development environment and run:

```sh
uv sync --locked
uv run pytest
```

## Research direction

The [design proposal](docs/design-proposal.md) describes the planned
`ModelVariant × TrainingMethod × ExperimentRecipe` composition. RT-DETRv3's
auxiliary branches, RT-DETRv4-style distillation, DEIM recipes, and D-FINE-style
distribution refinement inform these boundaries. They are design targets, not
implemented integrations. New regression representations may replace an entire
decoder/selector boundary rather than conforming to RT-DETR coordinate deltas.

See the [implementation roadmap](docs/roadmap.md) and the
[extension example](docs/extensions.md) for the next concrete steps.

## License

Clean RT-DETR is licensed under the [Apache License, Version 2.0](LICENSE).
The vendored RT-DETR source and adaptations retain the upstream Apache-2.0
license and attribution notices described in [third_party/RT-DETR/README.md](third_party/RT-DETR/README.md).

## Provenance

Based on [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), pinned to commit
`068dfde65f2667ad6555883c69d73de886518cad`. Upstream-derived modules carry change
notices. The [oracle source and license](third_party/RT-DETR/README.md) are retained
for parity testing. [Baseline metadata](benchmarks/baseline.json) explicitly marks
checkpoint, AP, and latency measurements as pending.
