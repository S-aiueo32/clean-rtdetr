# Documentation

English | [日本語](ja/README.md)

Use these guides to run inference, evaluate predictions, train a model, or extend
Clean RT-DETR. They assume Python and PyTorch knowledge. For a CPU example with
no external artifacts, start with the [quick start](../README.md#quick-start).

## Choose a guide

| Task | Guide |
| --- | --- |
| Convert weights, predict from images, or export ONNX | [Inference and export](inference.md) |
| Generate COCO prediction JSON and calculate bbox metrics | [COCO evaluation](evaluation.md) |
| Prepare targets and use matching, losses, or denoising queries | [Training losses](training.md) |
| Configure a training run and resume at an epoch boundary | [Lightning training](lightning.md) |
| Check CUDA, AMP, and DDP execution limits and measured results | [GPU validation](gpu-validation.md) |
| Change model components or tensor contracts | [Extension guide](extensions.md) |
| Build a research recipe using public package APIs | [Recipe responsibilities](recipe-boundaries.md) and [recipes](../recipes/README.md) |
| Develop, test, or build the packages | [Contributing](../CONTRIBUTING.md) |
| Check source, weight, dataset, or checkpoint license scope | [Licensing](licensing.md) |

## Prepare the environment

Run the guides from the repository root with Python 3.13 or 3.14 and uv.
Choose the setup for your task:

| Task | Setup command |
| --- | --- |
| Develop and test all three packages | `uv sync --all-packages` |
| Use core tensor APIs | `uv sync --package clean-rtdetr` |
| Read images and generate detections | `uv sync --package clean-rtdetr --extra inference` |
| Calculate COCO metrics from saved predictions | `uv sync --package clean-rtdetr --extra evaluation` |
| Use image inference and COCO metrics together | `uv sync --package clean-rtdetr --extra inference --extra evaluation` |
| Export ONNX and check CPU runtime parity | `uv sync --package clean-rtdetr-export` |
| Use matching, losses, denoising, or Lightning | `uv sync --package clean-rtdetr-training` |

Package-specific sync commands can remove other members or extras from the
shared `.venv`. Repeat setup when switching tasks, and restore
`uv sync --all-packages` before full-workspace validation. The `evaluation`
extra is optional and is not included by `--all-packages`. Subsequent `uv run`
commands use the installed environment; repeat sync after changing dependencies.
Recipes have their own environments and setup instructions.

Image prediction and checkpoint-backed export need a local native state dict;
the [inference guide](inference.md) explains conversion with
`clean-rtdetr-convert`. COCO prediction also needs local images and annotations.
The commands do not download these artifacts.

## Check project status

The [roadmap](roadmap.md) separates implemented features, validation evidence,
and remaining work. It links the CPU parity reports, historical pretrained COCO
evaluation, and bounded GPU checks. Use it to assess support for your workload.

The [design proposal](design-proposal.md) explains the intended architecture and
future deployment work. Proposed APIs and acceptance targets are not implemented
features or measured results.
