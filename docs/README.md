# Documentation

English | [日本語](ja/README.md)

Clean RT-DETR separates the core detector, ONNX export, and training into three
packages. Start with the [repository quick start](../README.md#quick-start) for
a synthetic CPU example that needs no external artifacts.

## Choose an environment

Run the commands below from the repository root with Python 3.13 or 3.14 and uv.
Choose the environment for your task:

| Task | Setup command |
| --- | --- |
| Develop and test all three packages | `uv sync --all-packages` |
| Run core tensor APIs | `uv sync --package clean-rtdetr` |
| Read images and generate detections | `uv sync --package clean-rtdetr --extra inference` |
| Calculate COCO metrics from saved predictions | `uv sync --package clean-rtdetr --extra evaluation` |
| Use image inference and COCO metrics together | `uv sync --package clean-rtdetr --extra inference --extra evaluation` |
| Export ONNX and verify CPU runtime parity | `uv sync --package clean-rtdetr-export` |
| Use matching, losses, denoising, or Lightning | `uv sync --package clean-rtdetr-training` |

Package-specific sync commands can remove other members or extras from the
shared `.venv`. Re-run the setup for your next task when switching guides;
restore `uv sync --all-packages` before full-workspace validation.
The evaluation extra is optional and is not included by `--all-packages`.
Package and extra selection happens during `uv sync`; subsequent guide commands
use `uv run` with the installed environment. Re-run `uv sync` when package
metadata or dependency selections change.

## User guides

- [Inference I/O](inference.md): model and image contracts, a Python inference
  example, checkpoint loading, postprocessing, and fixed-shape ONNX execution.
- [COCO evaluation](evaluation.md): local dataset requirements, prediction JSON,
  category mapping, subset evaluation, and metric reports.
- [Training losses](training.md): target format, Hungarian matching, weighted
  losses, denoising queries, and gradient/parity guarantees.
- [Lightning training](lightning.md): synthetic training, Hydra configuration,
  optimizer/EMA state, and epoch-boundary resume.

Image inference and checkpoint-backed export require a local converted native
state dict. The installed `clean-rtdetr` package includes the
`clean-rtdetr-convert` command for the pinned official R50vd envelope; the
checkout-only `tools.compatibility` helpers remain available for full, stage,
and training compatibility work.
COCO prediction also requires local images and annotations. These commands do
not download those artifacts; checkpoint provenance is recorded in
[baseline metadata](../benchmarks/baseline.json).

## Development and design

- [Contributing](../CONTRIBUTING.md): workspace layout, branch conventions,
  required validation, and distribution builds.
- [Extension guide](extensions.md): component injection, tensor annotations,
  and gradient ownership.
- [Implementation roadmap](roadmap.md): completed work and remaining validation.
- [Design proposal](design-proposal.md): intended research architecture and
  future integrations; it is not a list of supported features.

## Validation evidence

The [full-model report](../benchmarks/full-model-parity.json) compares official
R50vd weights on a seeded synthetic input. The
[ONNX report](../benchmarks/onnx-parity.json) records fixed batch-1 640×640 CPU
FP32 parity with graph optimizations disabled. Training tests cover CPU losses,
gradients, and synthetic epoch-boundary resume. These checks do not establish
full-dataset COCO AP, convergence, GPU/DDP/AMP behavior, or latency.
