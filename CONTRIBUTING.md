# Contributing

## Development setup

Python 3.12 is used for development. Runtime dependencies are PyTorch and
jaxtyping; pytest, Ruff, and ty are in the development group. Use the committed
lockfile:

```sh
uv sync --locked
```

## Validation

Run the test suite, linters, type checker, and example with:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python examples/query_selection.py
```

Ruff selects `ALL`, with documented exceptions for formatter/docstring
conflicts, pytest assertions and upstream private-method access, and example
output. The hash-verified oracle loader has one local `S102` exemption for
executing the pinned methods. The three modules with jaxtyping shape strings
exempt `F722` because these strings are not Python forward references; `F821`
remains enabled. ty checks `src`, `tests`, and `examples` without rule
suppressions. The unchanged vendored upstream source is outside both tools'
check scope.

## Implementation notes

Validators currently use eager checks and host-visible shape metadata. Export
will have a separate tensor-only boundary; this slice is not export-ready.

Tensor boundaries use annotations such as `Float[Tensor, "batch queries 4"]` and
`Int64[Tensor, "batch queries"]`. These document tensor shape/dtype while ty checks
Python types; they do not statically prove dimension equality. No automatic
jaxtyping runtime decorators are installed: existing explicit validators remain
responsible for runtime checks. JAX is not required. See the
[tensor annotation conventions](docs/extensions.md#tensor-annotation-conventions).
