# RT-DETRv4 recipe provenance

Reference repository: [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)

Pinned reference commit: `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`.

Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
Upstream RT-DETRv4 is Apache-2.0; the license text is included in [LICENSE](LICENSE).
Recipe implementations are written against Clean RT-DETR's public interfaces.

Reference behavior:

- [hybrid_encoder.py](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/hybrid_encoder.py):
  post-AIFI F5 and pointwise Linear projection into teacher channels.
- [dinov3_teacher.py](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/dinov3_teacher.py):
  ImageNet normalization, average pooling by two, frozen normalized patch tokens.
- [rtv4_criterion.py](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/rtv4_criterion.py):
  teacher interpolation, token normalization and mean one-minus-cosine.
  The implementation includes +1 compared with the paper's negative cosine.
- [det_engine.py](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/solver/det_engine.py) and
  [det_solver.py](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/solver/det_solver.py):
  GAM measures AIFI's percentage of total gradient L1 magnitude. Outside the band
  it targets the opposite boundary: observed fraction p and target fraction q
  produce factor q(1-p)/(p(1-q)), bounded to [0.1, 10]. Zero resets the weight.

Differences and validation scope:

- Student uses Clean RT-DETRv2 R18vd with baseline detection and DN losses.
  This is not the complete official RT-DETRv4 detector family.
- AIFI and distillation features are explicit outputs, without hidden caches.
- GAM uses single-process CPU FP32 pre-clipping gradients. AIFI parameters are
  selected by module identity. The denominator includes all trainable parameters,
  including the DSI projector, DN and detection components.
- GAM is a Lightning callback with the mutable weight in a criterion buffer.
  `stop_epoch` is optional; upstream augmentation schedules are not reproduced.
- Missing/invalid training features raise errors instead of producing a CUDA zero loss.
- Rectangular stride-aligned teacher token grids are supported explicitly.
- DINOv3 loads caller-provided local code and weights and is not vendored.
  Its [separate license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) applies.
  `SmokeTeacher` is random test code with no pretrained semantic knowledge.
- Tests validate this composition, not full detector parity, pretrained DINOv3
  execution, convergence, COCO AP or latency.
