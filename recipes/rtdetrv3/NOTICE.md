# Method provenance and scope

Reference: [RT-DETRv3: Real-Time End-to-End Object Detection with Hierarchical
Dense Positive Supervision (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.html),
Shuo Wang, Chunlong Xia, Feng Lv, Yifeng Shi.

The official implementation is [clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3),
revision `349e7d99a5065e7b684118912e6a74178d4f4625`. It is Paddle-based and Apache-2.0 licensed. Its transformer
source credits Copyright (c) 2023 PaddlePaddle Authors, Copyright (c) 2020
SenseTime, and Copyright 2022 The IDEA Authors. Repository-level provenance is
recorded in [third_party/RT-DETR/rtdetrv3-recipe.md](../../third_party/RT-DETR/rtdetrv3-recipe.md).

This recipe is a new PyTorch composition of Clean RT-DETR's public stages.
It does not import or vendor Paddle code. The local recipe is Apache-2.0 licensed;
see [LICENSE](LICENSE).

The following behavior follows the official transformer implementation:

- `ppdet/modeling/transformers/rtdetr_transformerv3.py`: independent selector
  heads, shared mapped detached memory for extra groups, shared decoder and heads,
  random SAP edge masks with probability 0.1, and isolated group attention.
  Paddle's `True=allow` convention is inverted for PyTorch's `True=block`.
- `ppdet/modeling/heads/detr_head.py` and `ppdet/modeling/losses/detr_loss.py`:
  average O2O group losses and use four repeated copies of each GT for O2M
  Hungarian assignment and normalization. Every retained decoder layer and
  encoder prediction receives a loss. O2M has no DN queries.

Differences from the official full recipe:

- The CNN auxiliary branch uses a small Conv/SiLU head and Task-Aligned
  assignment throughout. It is not PP-YOLOE's ESE head and does not implement
  its initial ATSS schedule. Its regression uses GIoU and DFL. Decoded CNN box
  corners are clipped to the image before conversion to the library's normalized
  cxcywh contract. Classification and regression normalize by summed assigned
  quality, clamped to one; regression terms are quality-weighted.
- SAP groups execute sequentially against shared memory instead of concatenating
  all groups into a single block-masked decoder input. With the default zero
  dropout, isolation has the same attention structure; no full upstream numeric
  parity is claimed. Fully blocked SAP rows retain their diagonal to avoid NaNs
  in small-query tests, a deliberate robustness difference.
- Baseline Clean RT-DETR owns model initialization, VFL/L1/GIoU, and DN semantics.
  DN defaults to four rather than the official larger training setting.
- CPU FP32, synthetic/Penn-Fudan data, random weights, and a one-epoch smoke
  schedule replace official COCO training, augmentation, and pretrained weights.

These components demonstrate caller-owned training extensions. They do not
establish official checkpoint compatibility, full-dataset AP, convergence,
latency, or export support.
