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

1. Add PResNet and HybridEncoder, then extend strict checkpoint conversion to complete R50vd.
2. Bring up PResNet and HybridEncoder, including an explicit AIFI feature boundary.
3. Establish complete-model checkpoint/stage parity before claiming an inference baseline.
4. Add fixed-shape export, then DN, criterion, training/resume and composition.

## Research acceptance

Exercise training-only branches with shared decoder weights, feature distillation
and controller state, distribution regression, and recipe/loss composition. Each
extension progresses independently through contract tests, reproduction, export,
and backend verification. Start official DEIM + D-FINE and RT-DETRv3/v4 integrations
once their required boundaries have passed these smaller checks.
