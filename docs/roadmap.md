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
4. [ ] Add preprocessing/postprocessing and capture COCO AP in a fixed evaluation environment.
5. [ ] Add fixed-shape export, then DN, criterion, training/resume and composition.

The image-to-prediction API returns raw class logits and normalized cxcywh boxes.
Small offline tests cover component integration, gradients, and checkpoint
validation; `examples/checkpoint_parity.py --full-model` checks the official
weights on a seeded synthetic 640×640 input. That comparison is separate from
dataset evaluation, GPU fixture capture, and latency measurement.
The [recorded parity result](../benchmarks/full-model-parity.json) includes source
hashes, checkpoint provenance, input hash, environment, and per-stage errors.

## Research acceptance

Exercise training-only branches with shared decoder weights, feature distillation
and controller state, distribution regression, and recipe/loss composition. Each
extension progresses independently through contract tests, reproduction, export,
and backend verification. Start official DEIM + D-FINE and RT-DETRv3/v4 integrations
once their required boundaries have passed these smaller checks.
