# Agent instructions

## Project overview

Clean RT-DETR is a readable PyTorch reference implementation. The current
scope covers multi-scale features, projected memory, and selected queries; it
is not yet a complete detector.

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
uv run python examples/query_selection.py
```

Ruff and ty configuration lives in `pyproject.toml`. The vendored upstream
directory is excluded from those checks.
