# Implementation roadmap

English | [日本語](ja/roadmap.md)

Use this page to distinguish implemented features from measured results and
planned work. The [design proposal](design-proposal.md) describes the broader
architecture target.

## Current implementation

The inference path connects images → backbone → HybridEncoder → projected
memory → selected queries → iterative decoder predictions. Core, export, and
training are separate distributions; export and training each depend on core.

| Area | Implemented scope | Details |
| --- | --- | --- |
| Model composition | Component-injected `RTDETR`, ten RT-DETR/RT-DETRv2 PResNet presets, explicit decoder execution/output plans, optional AIFI features | [Extensions](extensions.md) |
| Inference | Official R50vd inference-weight conversion, native state-dict loading, Pillow/uint8 preprocessing, top-k postprocessing | [Inference](inference.md) |
| Evaluation | COCO prediction JSON, category-ID mapping, optional bbox metrics | [COCO evaluation](evaluation.md) |
| Export | `FixedShapeRTDETR`, fixed batch-1 640×640 CPU FP32 ONNX Runtime execution with graph optimizations disabled | [Export profile](inference.md#fixed-shape-onnx-export) |
| Training losses | Hungarian matching, Varifocal/L1/GIoU, decoder/encoder auxiliary supervision, global target-count normalization under native DDP | [Training losses](training.md) |
| Denoising | Contrastive queries, registered class embeddings, query separation, per-layer DN losses | [Denoising](training.md#contrastive-denoising) |
| Training loop | Lightning, AdamW parameter groups, EMA, CPU FP32/`bf16-mixed` and CUDA FP32/AMP, epoch-boundary resume | [Lightning](lightning.md) and [GPU scope](gpu-validation.md) |
| Research examples | DEIM MAL, D-FINE FDR, RT-DETRv3 training branches, RT-DETRv4 DSI/GAM compositions | [Recipes](../recipes/README.md) |

## Validation evidence

Reports apply to their recorded source revision, inputs, and environment.
Passing a synthetic check does not establish full-dataset training or accuracy.

| Evidence | What it establishes |
| --- | --- |
| [CPU full-model parity](../benchmarks/full-model-parity.json) | Official R50vd weights compared on a seeded synthetic 640×640 input, with checkpoint and source hashes and per-stage errors |
| [CPU ONNX parity](../benchmarks/onnx-parity.json) | Fixed batch-1 640×640 FP32 execution for seeds 41 and 42, with exporter and runtime graph optimizations disabled |
| [Training tests](training.md#validation-scope) and [resume tests](lightning.md#validation) | CPU assignment, loss, gradient, and DN parity against pinned upstream code; synthetic checkpoint/resume checks |
| [COCO val2017 evaluation](../benchmarks/coco-val2017.md) | Historical official pretrained R50vd EMA inference: CPU FP32 bbox AP 53.05 on all 5,000 images; this is pretrained-weight evaluation |
| [GPU validation](gpu-validation.md) | Bounded CUDA precision, training, resume, and native DDP checks with their recorded limits |
| [Historical L4 bf16 probe](../benchmarks/bf16-l4.md) | Random-weight R50 backbone with small encoder/decoder at 64×64, plus raw DN training on that snapshot; it does not validate later Lightning GPU support |

## Remaining work

- Capture full R50vd stage fixtures, COCO AP, and latency in a fixed GPU environment.
- Compare full-dataset COCO results with the upstream implementation under
  identical conditions.
- Validate full-dataset training, convergence, and training-derived COCO AP.
- Reproduce official research methods, checkpoints, and training schedules.
- Validate deployment beyond the fixed CPU ONNX profile, including dynamic
  shapes, split export, and TensorRT.

Short GPU/DDP/AMP runs do not complete full training validation. Historical
pretrained COCO results do not complete independent training reproduction.

## Research acceptance

Each extension progresses independently through contract tests, reproduction,
export verification, and backend verification. The current recipes demonstrate
composition and bounded numerical or integration checks; they do not establish
official DEIM + D-FINE or RT-DETRv3/v4 reproduction.

Before claiming a new stage of support, record its configuration, seed, source
revision, artifacts, and comparison results. Use the
[recipe responsibilities](recipe-boundaries.md) to decide which contracts,
losses, data transformations, and state belong to the extension.
