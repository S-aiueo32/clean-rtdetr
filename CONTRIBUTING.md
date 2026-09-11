# Contributing

[日本語版](CONTRIBUTING.ja.md)

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

Python 3.13 and 3.14 are supported; local development defaults to 3.14. The uv
workspace contains the `clean-rtdetr`, `clean-rtdetr-export`, and
`clean-rtdetr-training` distributions. Export and training depend on core, but
not on each other. Each distribution owns a `packages/<member>/pyproject.toml`,
README, source tree, and test tree. The root `pyproject.toml` is workspace and
shared-tool configuration, not publishable package metadata. Runtime
dependencies belong to the package that uses them; beartype, pytest,
pytest-cov, Hypothesis, Ruff, and ty are development tools.
Pillow is available through the optional `inference` extra and is also installed
for development image tests. Core model imports and postprocessing remain
independent of Pillow. Install the complete development workspace:

```sh
uv sync --all-packages
```

The base core distribution includes the internal `clean-rtdetr-convert` CLI and
does not require Pillow, export, or training dependencies for checkpoint
conversion. The public installed surface contains no compatibility API; full,
stage, and training compatibility helpers under `tools.compatibility` are
checkout-only.

`uv sync` uses the repository's `uv.lock` file and may update it when project
dependencies change. When changing package dependencies, edit the owning member
`pyproject.toml`, run `uv lock`, and include the resulting `uv.lock` change. Keep shared
experiment-specific tools in the root `dev` dependency group unless a member
requires them for its own development or runtime.

Install the repository hooks once per checkout:

```sh
uv run pre-commit install
```

The hooks apply Ruff fixes and run ty before each commit. Run them manually on
all tracked files with
`uv run pre-commit run --all-files`.
The full pytest
suite remains a CI/handoff check because it is more expensive than a commit
hook.

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

The test command also emits a coverage report. New behavior should include
focused tests. The suite checks contracts and parity; passing it does not
establish dataset accuracy or training convergence. Before handoff, also run the
builds and metadata dry runs in [Distribution artifacts](#distribution-artifacts).

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

CI runs the same checks on Python 3.13 and 3.14. Package-boundary jobs select
members explicitly. CI passes `--locked` to verify that the committed `uv.lock`
is current. For example, the export job uses:

```sh
uv sync --locked --package clean-rtdetr-export
uv run pytest \
  packages/export/tests/test_export.py packages/export/tests/test_export_cli.py
```

To conserve GitHub Actions usage, automatic push runs are limited to `main`.
Pull request checks remain enabled, and all five workflows can also be started
manually with `workflow_dispatch` once the workflow exists on the default
branch. Each workflow has its own concurrency group. On `main`, the group also
includes the run ID, so pending and running `main` runs are never superseded.
For pull requests the group includes the pull request number, and for other
branches it includes the ref; those groups use `cancel-in-progress: true`.

A local pass does not justify
claiming complete-detector AP, latency, checkpoint, or export support; those
claims require the corresponding implementation and fixtures.

Ruff selects `ALL`, with documented exceptions for formatter/docstring
conflicts, pytest assertions and upstream private-method access, and example
output. The hash-verified oracle loader has one local `S102` exemption for
executing the pinned methods. Modules with jaxtyping shape strings
exempt `F722` because these strings are not Python forward references; `F821`
remains enabled. ty checks package-local `src` and `tests` trees plus `tools`
without rule suppressions. The unchanged vendored upstream source is outside
both tools' check scope.

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

Inspect the resulting archives to ensure each contains only its own Python
namespace, package data, README, license, and provenance files. The public
namespaces are `clean_rtdetr`, `clean_rtdetr_export`, and
`clean_rtdetr_training`; do not restore `clean_rtdetr.export` or
`clean_rtdetr.training` compatibility shims. Release versions must satisfy the
member dependency constraints, and `clean-rtdetr` must be available on the
target index before publishing export or training.

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
