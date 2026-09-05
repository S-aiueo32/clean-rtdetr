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

- [x] Add PResNet and HybridEncoder, including a differentiable AIFI boundary.
- [x] Connect image inputs to decoder predictions and load every inference-stage checkpoint tensor.
- [x] Validate CPU FP32 image-to-prediction stage parity at 640×640 with the official EMA checkpoint.

1. Add preprocessing/postprocessing and capture full R50vd fixtures, COCO AP, and
   latency in a fixed GPU environment before claiming an evaluated inference baseline.
2. Add fixed-shape export, then DN, criterion, training/resume and composition.

PResNet, HybridEncoder (including the AIFI boundary), and strict R50vd model
checkpoint loading are implemented. The model API is `build_rtdetr_r50vd()` and
accepts RGB float tensors in `[0, 1]` without mean/std normalization;
preprocessing remains outside the package. Stage parity does not establish COCO AP.

## Research acceptance

Exercise training-only branches with shared decoder weights, feature distillation
and controller state, distribution regression, and recipe/loss composition. Each
extension progresses independently through contract tests, reproduction, export,
and backend verification. Start official DEIM + D-FINE and RT-DETRv3/v4 integrations
once their required boundaries have passed these smaller checks.
