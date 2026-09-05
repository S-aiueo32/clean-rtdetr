# Clean RT-DETR

A readable RT-DETR reference implementation with composable research extensions.
The design separates model variants, training methods, and experiment recipes
from deployment transformations.

## Current implementation

The first slice implements **multi-scale features → projected memory → selected
queries** in PyTorch, with constructor injection for score and box heads.

- Frozen feature/query contracts with eager boundary validation.
- Decoder input projection, row-major flattening, and explicit level offsets.
- Anchor generation and top-k selection preserving upstream masking semantics.
- Feature-content detach and differentiable encoder auxiliary predictions.
- Forward and gradient parity tests against SHA-256-pinned upstream methods.

This is not yet a complete detector. Backbone, HybridEncoder, decoder, denoising,
training criteria, official checkpoint conversion, COCO evaluation, and export
remain to be implemented. No AP, latency, or complete-model parity is claimed.
The included example uses synthetic features and randomly initialized weights.

## Development

Python 3.12 is used for development. The runtime dependency is PyTorch; pytest,
Ruff, and ty are in the development group. Use the committed lockfile:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python examples/query_selection.py
```

Ruff selects `ALL`, with documented exceptions for formatter/docstring conflicts,
pytest assertions and upstream private-method access, and example output. The
hash-verified oracle loader has one local `S102` exemption for executing the
pinned methods. ty checks `src`, `tests`, and `examples` without rule suppressions.
The unchanged vendored upstream source is outside both tools' check scope.

Validators currently use eager checks and host-visible shape metadata. Export
will have a separate tensor-only boundary; this slice is not export-ready.

## Research direction

The [design proposal](docs/design-proposal.md) describes the planned
`ModelVariant × TrainingMethod × ExperimentRecipe` composition. RT-DETRv3's
auxiliary branches, RT-DETRv4-style distillation, DEIM recipes, and D-FINE-style
distribution refinement inform these boundaries. They are design targets, not
implemented integrations. New regression representations may replace an entire
decoder/selector boundary rather than conforming to RT-DETR coordinate deltas.

See the [implementation roadmap](docs/roadmap.md) and the
[extension example](docs/extensions.md) for the next concrete steps.

## Provenance

Based on [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), pinned to commit
`068dfde65f2667ad6555883c69d73de886518cad`. Upstream-derived modules carry change
notices. The [oracle source and license](third_party/RT-DETR/README.md) are retained
for parity testing. [Baseline metadata](benchmarks/baseline.json) explicitly marks
checkpoint, AP, and latency measurements as pending.
