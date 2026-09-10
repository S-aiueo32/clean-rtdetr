# Implementation roadmap

English | [日本語](ja/roadmap.md)

The design proposal is an architectural target. This file records implemented
scope separately so a passing small-stage test cannot imply complete reproduction.
The uv workspace publishes core, fixed-shape export, and training as separate
distributions with one-way dependencies on `clean-rtdetr`.

## Current implementation

The PyTorch inference path connects images → R50vd backbone → HybridEncoder →
projected memory → selected queries → iterative decoder predictions.

| Area | Implemented and validated scope | Details |
| --- | --- | --- |
| Model stages | Frozen tensor contracts with jaxtyping annotations and eager validation; row-major memory projection and level offsets; anchors and top-k selection preserving upstream masks | [Extension guide](extensions.md) |
| Decoder and gradients | Explicit execution/output plans, self-attention, multi-scale deformable cross-attention, iterative refinement, content/reference detach boundaries, differentiable encoder auxiliary predictions | [Gradient ownership](extensions.md#gradient-ownership) |
| Composition | Component-injected `RTDETR`, named architecture factories for ten official RT-DETR/RT-DETRv2 PResNet presets, optional AIFI features, and hierarchical Hydra configs for model and training components | [Extension guide](extensions.md#named-model-presets) |
| Checkpoints and inference | Strict official inference-weight loading, CPU FP32 full-model parity, Pillow/uint8 preprocessing, and upstream-compatible top-k postprocessing | [Inference I/O](inference.md) |
| COCO evaluation | Prediction JSON and an optional bbox metric adapter, tested on synthetic annotations; pycocotools is loaded only for metrics | [COCO evaluation](evaluation.md) |
| ONNX export | `FixedShapeRTDETR` and CPU FP32 ONNX Runtime parity for fixed batch-1 640×640, with graph optimizations disabled | [Export profile](inference.md#fixed-shape-onnx-export) |
| Matching and losses | Hungarian matching and Varifocal/L1/GIoU losses with distributed target-count normalization, including decoder/encoder auxiliary supervision and CPU loss/gradient parity | [Training losses](training.md) |
| Denoising | Contrastive queries, registered class embeddings in the training wrapper, and per-layer DN losses with CPU upstream parity | [Contrastive denoising](training.md#contrastive-denoising) |
| Training loop | Lightning adapter with baseline AdamW groups, EMA, and CPU FP32 / CUDA FP32 or AMP synthetic epoch-boundary resume | [Lightning training](lightning.md) |

Forward and gradient parity tests use SHA-256-pinned upstream methods.
Full-dataset training, research-method composition,
measured full-dataset COCO AP, and latency remain pending. Dynamic-shape ONNX
and TensorRT are outside the validated export scope.

## First increment

- [x] Private repository, Python package, isolated development environment.
- [x] Preserve the design proposal and pin upstream source provenance.
- [x] Feature contracts, memory projection, anchors, and query selection.
- [x] Score/box head injection with real parameter registration and gradients.
- [x] Compare stage outputs and gradients against unchanged upstream methods.
- [x] Query assembly and decoder with iterative refinement and explicit layer selection.
- [x] Capture an official checkpoint, its hash, and the model/EMA selection for implemented stages.
- [ ] Capture full R50vd stage fixtures, COCO AP, and latency in a fixed GPU environment.

## Next increment

1. [x] Add PResNet and HybridEncoder, including an explicit AIFI feature boundary.
2. [x] Connect image-to-prediction R50vd and extend strict checkpoint loading to all inference stages.
3. [x] Validate CPU FP32 complete-model checkpoint/stage parity with the pinned official artifact.
4. [x] Add Pillow/uint8 preprocessing, top-k postprocessing, and an image inference example.
5. [ ] Capture COCO AP in a fixed evaluation environment, including category-ID mapping.
   The prediction JSON and optional metric adapter are implemented and tested
   on synthetic annotations; full-dataset accuracy measurement remains pending.
6. [x] Add fixed-shape ONNX export and CPU FP32 runtime parity.
7. [x] Add DN queries, criterion, and CPU loss/gradient parity.
8. [x] Add synthetic CPU FP32 Lightning training and epoch-boundary resume.
9. [ ] Validate full-dataset training. Short synthetic GPU/DDP/AMP checks are described in [GPU execution](gpu-validation.md).
10. [ ] Implement research-method composition.

## Validation evidence

The FP32/FP64 matching criterion is implemented with Hungarian
assignment and Varifocal/L1/GIoU losses, including decoder and encoder auxiliary
supervision. Contrastive DN construction, registered training-wrapper embeddings,
and per-layer DN losses are implemented. CPU upstream query/loss/gradient parity
is covered by offline tests; see [training losses](training.md). The optional
[Lightning adapter](lightning.md) adds a CPU FP32 / CUDA FP32 or AMP training loop and epoch-boundary
resume, including optimizer, scheduler, EMA, and application RNG state. Hierarchical
Hydra configs inject baseline model stages and optimizer/scheduler factories. Full
dataset training and research-method composition remain pending. See [GPU execution](gpu-validation.md) for bounded GPU checks.

The image-to-prediction API returns raw class logits and normalized cxcywh boxes.
Separate image I/O accepts Pillow images or CPU uint8 RGB tensors using the same
640×640 Pillow resize path. Postprocessing returns scores, class indices, and
unclipped original-image xyxy boxes. Offline tests cover image I/O contracts and
pinned upstream postprocessor parity; they do not measure dataset accuracy.
Small offline tests cover component integration, gradients, and checkpoint
validation; `tools/checkpoint_parity.py --full-model` checks the official
weights on a seeded synthetic 640×640 input. That comparison is separate from
dataset evaluation, GPU fixture capture, and latency measurement.
The [recorded parity result](../benchmarks/full-model-parity.json) includes source
hashes, checkpoint provenance, input hash, environment, and per-stage errors.
The [fixed-shape export result](../benchmarks/onnx-parity.json) records a separate
CPU FP32 ONNX Runtime gate for seeds 41 and 42, with exporter and runtime graph
optimizations disabled. This does not complete the broader deployment phase or establish AP,
latency, dynamic-shape, or TensorRT support.

## Research acceptance

Exercise training-only branches with shared decoder weights, feature distillation
and controller state, distribution regression, and recipe/loss composition. Each
extension progresses independently through contract tests, reproduction, export,
and backend verification. Start official DEIM + D-FINE and RT-DETRv3/v4 integrations
once their required boundaries have passed these smaller checks.
