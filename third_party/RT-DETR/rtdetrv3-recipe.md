# RT-DETRv3 recipe provenance

The RT-DETRv3 recipe is informed by the official implementation at
<https://github.com/clxia12/RT-DETRv3>, pinned to commit
`349e7d99a5065e7b684118912e6a74178d4f4625` (2025-03-18). The upstream project
is Apache-2.0 licensed; see its
<https://github.com/clxia12/RT-DETRv3/blob/349e7d99a5065e7b684118912e6a74178d4f4625/LICENSE>.

Relevant upstream references are:

- `ppdet/modeling/transformers/rtdetr_transformerv3.py` (multi-group query
  attention masks and decoder groups)
- `ppdet/modeling/architectures/rtdetrv3.py` (training-only auxiliary branch)
- `ppdet/modeling/heads/detr_head.py` and
  `ppdet/modeling/losses/detr_loss.py` (O2M target replication and losses)
- `configs/rtdetrv3/_base_/rtdetrv3_r50vd.yml` (R50 recipe defaults)

This checkout implements a native PyTorch recipe. Its CNN auxiliary branch is
a local CNN/TaskAligned-loss substitute and is not a port of the upstream
PP-YOLOE head or its ATSS-to-TaskAlign training schedule. The provenance above
describes the design reference; it does not imply byte-for-byte or checkpoint
compatibility with PaddlePaddle RT-DETRv3.
