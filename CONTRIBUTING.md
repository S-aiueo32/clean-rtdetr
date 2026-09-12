# Contributing

[日本語版](CONTRIBUTING.ja.md)

Use this guide to set up the workspace, make a focused change, and run the
checks required before handoff. It assumes familiarity with Python, PyTorch,
and `uv`.

## Branch strategy

This repository uses lightweight trunk-based development. `main` should remain
validated, and changes should be merged through focused pull requests from
short-lived branches. There is no permanent `develop` branch.

Use these branch prefixes:

- `feat/<issue>-<short-name>` for new behavior
- `fix/<issue>-<short-name>` for bug fixes
- `refactor/<short-name>` for behavior-preserving restructuring
- `test/<short-name>` for test-only changes
- `docs/<short-name>` for documentation-only changes
- `experiment/<topic>` for temporary research experiments

Start from the latest `main`, do not push directly to `main`, and keep a pull
request focused on one purpose. Use squash merge and delete the branch after
merging. Research results must record their configuration, seed, and source
commit; an experiment branch name alone is not reproducibility metadata.

For releases, use version tags such as `v0.1.0` and `v0.1.1`. Add a release
branch only when a release requires independent stabilization or maintenance.

## Development setup

Python 3.13 and 3.14 are supported; local development uses 3.14 by default.
Install the complete workspace:

```sh
uv sync --all-packages
```

The workspace contains the independent `clean-rtdetr`, `clean-rtdetr-export`,
and `clean-rtdetr-training` distributions. Export and training depend on core,
not on each other. Each member owns its `pyproject.toml`, README, source, and
tests under `packages/<member>/`; the root `pyproject.toml` configures the
workspace and shared tools.

Put runtime dependencies in the member that uses them. When dependencies
change, update that member's `pyproject.toml`, run `uv lock`, and commit the
`uv.lock` update. Keep shared experiment-only tools in the root `dev` group.

The core distribution includes the private `clean-rtdetr-convert` CLI. Its
checkpoint conversion does not require Pillow, export, or training. Full,
stage, and training compatibility helpers in `tools/compatibility` are
checkout-only and are not a public installed API.

Install the repository hooks once per checkout:

```sh
uv run pre-commit install
```

The hooks apply Ruff fixes and run ty. Run `uv run pre-commit run --all-files`
for all tracked files; run the full pytest suite before handoff.

## Development smoke test

Run the model path with synthetic features and random weights; no image,
dataset, or checkpoint is required:

```sh
uv run python tools/query_selection.py
```

It prints the memory, selected-query, and decoder tensor shapes. This confirms
the execution path and shapes, not detection accuracy.

## Validation

Run the test suite, linters, type checker, and example with:

```sh
uv run pytest \
  --cov=clean_rtdetr --cov=clean_rtdetr_export --cov=clean_rtdetr_training \
  --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python tools/query_selection.py
```

The test command emits coverage. Add focused tests for new behavior. These
checks establish the tested contracts and parity, not dataset accuracy or
training convergence. Before handoff, also run the builds and metadata dry runs
in [Distribution artifacts](#distribution-artifacts).

During development, run a focused test with, for example:

```sh
uv run pytest \
  packages/core/tests/test_stages.py -k selection
```

Hypothesis tests should focus on cheap tensor-contract and geometry invariants
with bounded input sizes so the full suite remains suitable for CI. Local
artifacts such as `.pytest_cache/`, `.hypothesis/`, `.coverage`, and tool
caches must not be committed.

## Implementation workflow

For a new model stage:

1. Define or update the explicit tensor contract in
   `packages/core/src/clean_rtdetr/contracts.py`.
2. Implement core stages under `packages/core/src/clean_rtdetr/models`, export behavior under
   `packages/export/src/clean_rtdetr_export`, and training behavior under
   `packages/training/src/clean_rtdetr_training`.
3. Add focused contract, gradient, and shape tests under the owning member's
   `tests` directory.
4. Add parity tests against unchanged upstream code when the stage is derived
   from RT-DETR.
5. Update the roadmap and README when the supported scope changes.
6. Run the complete validation commands before handing off the change.

Keep upstream-derived source and provenance under `third_party/RT-DETR`.
When updating pinned upstream source, update its commit and SHA-256 in
`benchmarks/baseline.json` and `packages/core/tests/oracle.py`, retain license
and attribution notices, and re-run all parity tests. Do not rewrite the oracle
implementation into the package under test.

Core and training CI run on Python 3.13 and 3.14; export, evaluation, quality,
and distribution jobs use Python 3.13. CI uses `--locked` to verify `uv.lock`.
For example, the export job uses:

```sh
uv sync --locked --package clean-rtdetr-export
uv run pytest \
  packages/export/tests/test_export.py packages/export/tests/test_export_cli.py
```

CI uses separate core, training, export, evaluation, quality, distribution, and
recipe workflows. Native path filters select them on pull requests and pushes
to `main`; changed paths select the union of their checks. Each workflow can be
started with `workflow_dispatch` after it reaches the default branch. Superseded
runs are cancelled for a non-main ref; every `main` run is retained.

The filters follow these dependencies; multiple changed paths select the union
of the corresponding workflows:

| Changed files | Automatic checks |
| --- | --- |
| Core source | Core, training, export, evaluation, quality, core distribution, all recipes |
| Training source | Training, quality, training distribution, all recipes |
| Export source | Export, quality, export distribution |
| Package tests | Owning test workflow and quality; core COCO tests select evaluation |
| Root `configs/**` | Training |
| `tools/compatibility/**` | Relevant core/training tests and quality |
| `tools/query_selection.py` | Core and quality |
| Vendored `third_party/RT-DETR/**` | Core and training parity |
| `benchmarks/baseline.json` | Core checkpoint provenance tests |
| Package README/license/data | Owning distribution build |
| Recipe files except Markdown | Owning recipe; Python/tool configuration also selects quality |
| Root `pyproject.toml`, root `uv.lock`, or package `pyproject.toml` | All eight workspace workflows; core/training package metadata also selects recipes |
| Workflow YAML | That workflow |
| Root README, `docs/**` Markdown, CONTRIBUTING, AGENTS, other benchmark reports | None |

Workspace metadata selects all workspace workflows because locked operations
resolve the shared workspace. Recipes have independent locks and tool settings;
unrelated package tests, root-lock changes, and export changes do not select
them. Core changes select dependents conservatively. Distribution jobs are
separate from tests and static checks; the core job also checks the installed
conversion command outside the checkout.

When adding dependencies or moving files, update both push and pull-request
filters, including callers. A rename affects both paths. Use manual dispatch for
a full workflow run or a very large diff. A skipped workflow cannot be an
unconditional required check; see [GitHub's required-check guidance](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).

Ruff selects `ALL`, with documented exceptions for formatter/docstring conflicts,
pytest assertions, upstream private-stage access, CLI output, and jaxtyping
shape strings (`F722`). `F821` remains enabled. ty checks package `src` and
`tests` trees plus `tools` without suppressions. Unchanged vendored upstream
source is excluded from both tools.

A local pass does not establish complete-detector AP, latency, checkpoint, or
export support. Those claims need the corresponding implementation and fixtures.

## Distribution artifacts

The three members are independent PyPI distributions. Build both the sdist and
wheel for each member, then validate the metadata locally:

```sh
uv build --package clean-rtdetr --no-sources --out-dir dist/core
uv build --package clean-rtdetr-export --no-sources --out-dir dist/export
uv build --package clean-rtdetr-training --no-sources --out-dir dist/training
uv publish --dry-run dist/core/*
uv publish --dry-run dist/export/*
uv publish --dry-run dist/training/*
```

Inspect each archive for only its own Python namespace, package data, README,
license, and provenance. The public namespaces are `clean_rtdetr`,
`clean_rtdetr_export`, and `clean_rtdetr_training`; do not add
`clean_rtdetr.export` or `clean_rtdetr.training` shims. Release versions must
satisfy member constraints, and `clean-rtdetr` must be on the target index
before publishing export or training.

## Implementation notes

COCO prediction generation must work without pycocotools. Keep its imports and
foreign object types inside the private evaluation adapter; public interfaces
use project-owned records and plain metric dictionaries. Run the optional
metric tests with:

```sh
uv sync --package clean-rtdetr --extra evaluation
uv run pytest \
  packages/core/tests/test_coco.py packages/core/tests/test_coco_cli.py
```

The extra uses wheels only under the committed uv configuration. See
[evaluation](docs/evaluation.md).

Validators use eager checks and host-visible shape metadata. The fixed-shape
export adapter validates a copied model before capture, then reuses internal
attention, selection, and refinement arithmetic with specialized metadata.
Keep eager validation intact when changing those shared functions. See
[the export profile](docs/inference.md#fixed-shape-onnx-export) for supported
shapes and the required unoptimized CPU FP32 runtime settings.

Tensor boundaries use annotations such as `Float[Tensor, "batch queries 4"]` and
`Int64[Tensor, "batch queries"]`. ty checks Python types, while the pytest hook
uses jaxtyping with beartype to check annotated tensor shape/dtype at runtime.
Existing explicit validators remain responsible for cross-object invariants and
validation outside function annotations. JAX is not required. See the
[tensor annotation conventions](docs/extensions.md#tensor-annotation-conventions).
