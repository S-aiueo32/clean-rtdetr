# Clean RT-DETR

English | [日本語](README.ja.md)

A readable RT-DETR reference implementation with composable research extensions.
The design separates model variants, training methods, and experiment recipes
from deployment transformations.

See the [implementation roadmap](docs/roadmap.md) for implemented features,
validation evidence, and planned work.

## Quick start

Use Python 3.13 or 3.14 and uv. Run these commands from the repository root:

```sh
uv sync --all-packages
uv run python tools/query_selection.py
```

This small CPU example needs no images, datasets, or checkpoints. It uses random
weights to exercise feature projection, query selection, and decoding, and prints:

```text
memory: (2, 336, 32)
query content: (2, 20, 32)
encoder auxiliary boxes: (2, 20, 4)
decoder logits: (2, 20, 80)
decoder boxes: (2, 20, 4)
```

For detections on a real image, install `clean-rtdetr[inference]` and follow the
[inference guide](docs/inference.md) to download and convert the pinned official
checkpoint, then load the resulting native state dict. Random weights only
demonstrate the API.

## Documentation

| Task | Guide |
| --- | --- |
| Check implementation status and planned work | [Implementation roadmap](docs/roadmap.md) |
| Choose an environment or find the next guide | [Documentation index](docs/README.md) |
| Run image inference or load official weights | [Inference I/O](docs/inference.md) |
| Export and run an ONNX model | [Fixed-shape ONNX export](docs/inference.md#fixed-shape-onnx-export) |
| Generate COCO predictions and evaluate saved JSON | [COCO evaluation](docs/evaluation.md) |
| Work with targets, matching, and denoising losses | [Training losses](docs/training.md) |
| Run synthetic training and resume a checkpoint | [Lightning training](docs/lightning.md) |
| Extend model stages and understand gradient boundaries | [Extension guide](docs/extensions.md) |
| Develop, test, and build packages | [Contributing](CONTRIBUTING.md) |

## Packages

The uv workspace publishes three packages with one-way dependencies:

| Distribution | Python package | Workspace member | Responsibility |
| --- | --- | --- | --- |
| `clean-rtdetr` | `clean_rtdetr` | `packages/core` | Core model, inference, preprocessing, and evaluation |
| `clean-rtdetr-export` | `clean_rtdetr_export` | `packages/export` | Fixed-shape ONNX export and runtime parity |
| `clean-rtdetr-training` | `clean_rtdetr_training` | `packages/training` | Matching, losses, denoising, and Lightning training |

The export and training packages depend on the core package, but not on each
other. The former `clean_rtdetr.export` and `clean_rtdetr.training` namespaces
are not compatibility aliases. Each member has its own package metadata, README,
source tree, and tests so it can be built and published independently. The root
`pyproject.toml` coordinates the workspace and shared development tools; the
repository root is not itself a Python distribution.

## Contributing

Development setup, validation commands, linting policy, and implementation notes
are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

For a quick local check, install the development environment and run:

```sh
uv sync --all-packages
uv run pytest
```

See [distribution artifacts](CONTRIBUTING.md#distribution-artifacts) for package
builds, metadata dry runs, and release ordering.

## Research direction

The [design proposal](docs/design-proposal.md) describes the intended
`ModelVariant × TrainingMethod × ExperimentRecipe` architecture. See the
[extension guide](docs/extensions.md) for component boundaries and the
[implementation roadmap](docs/roadmap.md#research-acceptance) for research
acceptance criteria.

## License

Clean RT-DETR is licensed under the [Apache License, Version 2.0](LICENSE).
The vendored RT-DETR source and adaptations retain the upstream Apache-2.0
license and attribution notices described in [third_party/RT-DETR/README.md](third_party/RT-DETR/README.md).

## Provenance

Based on [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), pinned to commit
`068dfde65f2667ad6555883c69d73de886518cad`. Upstream-derived modules carry change
notices. The [oracle source and license](third_party/RT-DETR/README.md) are retained
for parity testing. [Baseline metadata](benchmarks/baseline.json) records the
official checkpoint URL, SHA-256, and EMA source. Validation reports and their
scope are listed in the [implementation roadmap](docs/roadmap.md).
