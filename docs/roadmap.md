# Implementation roadmap

The design proposal is an architectural target. This file records implemented
scope separately so a passing small-stage test cannot imply complete reproduction.

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
7. [ ] Add DN, criterion, training/resume, and composition.

The single-process FP32/FP64 matching criterion is implemented with Hungarian
assignment and Varifocal/L1/GIoU losses, including decoder and encoder auxiliary
supervision. Contrastive DN construction, registered training-wrapper embeddings,
and per-layer DN losses are implemented. CPU upstream query/loss/gradient parity
is covered by offline tests; see [training losses](training.md). Full
training/resume and composition remain pending, so step 7 is not complete.

The image-to-prediction API returns raw class logits and normalized cxcywh boxes.
Separate image I/O accepts Pillow images or CPU uint8 RGB tensors using the same
640×640 Pillow resize path. Postprocessing returns scores, class indices, and
unclipped original-image xyxy boxes. Offline tests cover image I/O contracts and
pinned upstream postprocessor parity; they do not measure dataset accuracy.
Small offline tests cover component integration, gradients, and checkpoint
validation; `examples/checkpoint_parity.py --full-model` checks the official
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
