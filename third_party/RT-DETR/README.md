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
comparing outputs and gradients. The independent full-stage oracle additionally
loads the official EMA checkpoint and compares image-to-prediction outputs. These
checks do not establish COCO AP, GPU latency, or an evaluated complete detector.

The production backbone, hybrid encoder, geometry, memory projection, and selection
modules are adapted from this upstream implementation. They contain change notices. This directory
is test-only and is not included in the installed Python wheel.

The pinned R50vd configuration and deformable-attention utility used by the
checkpoint parity oracle are retained under `rtdetr_pytorch/`. The official
checkpoint artifact is kept outside version control in `checkpoints/`; its URL,
SHA-256, and selected EMA state are recorded in `benchmarks/baseline.json`.

Additional unchanged sources from commit `068dfde65f2667ad6555883c69d73de886518cad`
support the independent backbone and hybrid encoder oracles:

- `src/nn/backbone/presnet.py` — `1faef37155a38d504e2ce3d569e09d8fbbf737c549d1fbb125cee9111fc7bc28`
- `src/nn/backbone/common.py` — `29ca34ee34f53ccab66e4a1d985df08ede3df9a686d4f84f861010b8f7513d44`
- `src/zoo/rtdetr/hybrid_encoder.py` — `3dff40011e125107fd1de2606429f04bc05d52a3d88e742b7ed655c5a07742c5`

The paths above are relative to `rtdetr_pytorch/`. Tests execute the original
classes and activation helpers after removing only registry decorators/imports.
`pretrained=False` prevents the upstream backbone from downloading weights.
