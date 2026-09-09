# Sources and scope

Based on [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution
Refinement](https://arxiv.org/abs/2410.13842), ICLR 2025.

Official source: [Peterande/D-FINE](https://github.com/Peterande/D-FINE/tree/956d1709314c2c6a4df6f34de232054578a7449f),
commit `956d1709314c2c6a4df6f34de232054578a7449f`.

- `src/zoo/dfine/dfine_utils.py`: weighting function, distance transforms, bin interpolation.
- `src/zoo/dfine/dfine_decoder.py`: FDR recurrence, initial reference detach, LQE.
- `src/zoo/dfine/dfine_criterion.py`: VFL, union assignments, FGL and DDF.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Upstream decoder is modified from RT-DETR, Copyright (c) 2023 lyuwenyu.
Apache-2.0; upstream `LICENSE` is included. Unmodified utility and criterion
sources are retained as `.txt` test fixtures under `tests/reference/`.
Tests load selected pure methods without importing the upstream registry.

Recipe changes: public Clean RT-DETR stage/training contracts; fixed Python
distribution constants; dtype-preserving bin interpolation; explicit prediction
payload; deterministic lexicographic tie-breaking in match unions; independent
checkpoint configuration; normalized targets but unclipped prediction boxes.
Default loss weights and temperature follow upstream. Only ordinary-query
single-process normalization is implemented. No DN, HGNetv2, D-FINE attention
gate, wider detachable layers, full training schedule, official weight loading,
full-model parity, AP or export claims are made.

The RT-DETR components and Hungarian cost layout come from Clean RT-DETR,
Copyright 2026 Clean RT-DETR contributors, based on `lyuwenyu/RT-DETR`
commit `068dfde65f2667ad6555883c69d73de886518cad` (Apache-2.0).
