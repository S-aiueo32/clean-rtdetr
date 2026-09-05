# Contributing

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

Python 3.13 and 3.14 are supported; local development defaults to 3.14.
Runtime dependencies are PyTorch and
jaxtyping; beartype, pytest, pytest-cov, Hypothesis, Ruff, and ty are in the
development group. Use the committed lockfile:

```sh
uv sync --locked
```

When changing dependencies, edit `pyproject.toml`, run `uv lock`, and include
the resulting `uv.lock` change. Keep experiment-specific tools in the `dev`
dependency group unless they are required by the package at runtime.

Install the repository hooks once per checkout:

```sh
uv run pre-commit install
```

The hooks apply Ruff fixes and run ty before each commit. Run them manually on
all tracked files with `uv run pre-commit run --all-files`. The full pytest
suite remains a CI/handoff check because it is more expensive than a commit
hook.

## Validation

Run the test suite, linters, type checker, and example with:

```sh
uv run pytest --cov=clean_rtdetr --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python examples/query_selection.py
```

The test command also emits a coverage report. New behavior should include
focused tests; a repository-wide coverage threshold will be introduced once
the complete detector stages are present.

During development, run a focused test with, for example:

```sh
uv run pytest tests/test_stages.py -k selection
```

Hypothesis tests should focus on cheap tensor-contract and geometry invariants
with bounded input sizes so the full suite remains suitable for CI. Local
artifacts such as `.pytest_cache/`, `.hypothesis/`, `.coverage`, and tool
caches must not be committed.

## Implementation workflow

For a new model stage:

1. Define or update the explicit tensor contract in `src/clean_rtdetr/core`.
2. Implement the stage under `src/clean_rtdetr/models`.
3. Add focused contract, gradient, and shape tests under `tests`.
4. Add parity tests against unchanged upstream code when the stage is derived
   from RT-DETR.
5. Update the roadmap and README when the supported scope changes.
6. Run the complete validation commands before handing off the change.

Keep upstream-derived source and provenance under `third_party/RT-DETR`.
When updating pinned upstream source, update its commit and SHA-256 in
`benchmarks/baseline.json` and `tests/oracle.py`, retain license and attribution
notices, and re-run all parity tests. Do not rewrite the oracle implementation
into the package under test.

CI runs the same checks on Python 3.13 and 3.14. A local pass does not justify
claiming complete-detector AP, latency, checkpoint, or export support; those
claims require the corresponding implementation and fixtures.

Ruff selects `ALL`, with documented exceptions for formatter/docstring
conflicts, pytest assertions and upstream private-method access, and example
output. The hash-verified oracle loader has one local `S102` exemption for
executing the pinned methods. Modules with jaxtyping shape strings
exempt `F722` because these strings are not Python forward references; `F821`
remains enabled. ty checks `src`, `tests`, and `examples` without rule
suppressions. The unchanged vendored upstream source is outside both tools'
check scope.

## Implementation notes

Validators currently use eager checks and host-visible shape metadata. Export
will have a separate tensor-only boundary; this slice is not export-ready.

Tensor boundaries use annotations such as `Float[Tensor, "batch queries 4"]` and
`Int64[Tensor, "batch queries"]`. ty checks Python types, while the pytest hook
uses jaxtyping with beartype to check annotated tensor shape/dtype at runtime.
Existing explicit validators remain responsible for cross-object invariants and
validation outside function annotations. JAX is not required. See the
[tensor annotation conventions](docs/extensions.md#tensor-annotation-conventions).
