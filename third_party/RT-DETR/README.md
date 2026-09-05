# Pinned upstream oracle

Source: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

- Commit: `068dfde65f2667ad6555883c69d73de886518cad`
- File: `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py`
- SHA-256: `c2837740cb66f128b72e802463db57ee3653e15d964baa3b9cf58cc27a817588`
- License: [Apache-2.0](LICENSE). The source attribution is retained in the file.

The decoder file is copied without modifications. Tests extract and execute the
original `_get_encoder_input`, `_generate_anchors`, and `_get_decoder_input`
methods through Python's AST, avoiding the upstream registry and optional training
dependencies. Head and projection weights are copied into this oracle before
comparing outputs and gradients. This validates extracted stages, not a complete
detector or an official trained checkpoint.

The production geometry, memory projection, and selection modules are adapted
from this upstream implementation. They contain change notices. This directory
is test-only and is not included in the installed Python wheel.

The pinned R50vd configuration and deformable-attention utility used by the
checkpoint parity oracle are retained under `rtdetr_pytorch/`. The official
checkpoint artifact is kept outside version control in `checkpoints/`; its URL,
SHA-256, and selected EMA state are recorded in `benchmarks/baseline.json`.
