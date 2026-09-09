# Agent instructions

## Project overview

Clean RT-DETR is a readable PyTorch reference implementation. The current
scope covers image-to-prediction R50vd, including multi-scale features, projected
memory, selected queries, and iterative decoder predictions, plus separate
Pillow/uint8 image preprocessing, top-k detection postprocessing, and fixed-shape
ONNX export with CPU FP32 runtime parity. Separate COCO prediction JSON and
optional bbox metric evaluation are implemented. Single-process Hungarian
matching and Varifocal/L1/GIoU training losses are implemented with CPU parity.
Contrastive denoising queries, a training wrapper, and DN losses have CPU parity.
An optional Lightning adapter supports CPU FP32 training and epoch-boundary resume.
Full training and measured full-dataset COCO AP remain outside the validated scope.

## Repository guidance

- Put package code under `src/clean_rtdetr`.
- Put regression and parity tests under `tests`.
- Keep upstream-derived code and provenance under `third_party/RT-DETR`.
- Preserve tensor contracts, validation behavior, and upstream parity semantics
  when changing model stages.
- Do not claim complete detector, AP, latency, checkpoint, or export support
  unless the corresponding implementation and validation exist.
- Update relevant documentation when changing public behavior or project scope.
- Delegate inexpensive, routine implementation work and command execution such
  as `fetch` to a lower-cost subagent when subagents are available. Keep
  architectural decisions, review, integration, and final verification with
  the primary agent.

## Branch strategy

Use lightweight trunk-based development. Keep `main` in a validated state and
merge short-lived branches through pull requests; do not create a permanent
`develop` branch.

- Name branches `feat/<issue>-<short-name>`, `fix/<issue>-<short-name>`,
  `refactor/<short-name>`, `test/<short-name>`, or `docs/<short-name>`.
- Use `experiment/<topic>` only for temporary research work. Record the
  configuration, seed, and source commit when reporting results; do not treat
  an experiment branch name as reproducibility metadata.
- Start branches from the latest `main`, keep each pull request focused, and
  do not push directly to `main`.
- Before handoff or merge, run the complete validation commands below. Use
  squash merge and delete merged branches.
- Use version tags such as `v0.1.0` and `v0.1.1` for releases. Avoid release
  branches until a release needs independent stabilization or maintenance.

## Development

Use the committed environment and lockfile:

```sh
uv sync --locked
```

Before handing off changes, run:

```sh
uv run pytest --cov=clean_rtdetr --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python tools/query_selection.py
```

Ruff and ty configuration lives in `pyproject.toml`. The vendored upstream
directory is excluded from those checks.
