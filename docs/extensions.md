# First extension boundary

English | [日本語](ja/extensions.md)

`QuerySelector` accepts ordinary `nn.Module` instances. Injected modules are
registered children, retain their initialization, participate in autograd, and
appear in `state_dict()` and `parameters()`.

```python
from torch import nn
from clean_rtdetr import QuerySelector, QuerySelectorConfig

selector = QuerySelector(
    QuerySelectorConfig(hidden_dim=256, num_classes=80, num_queries=300),
    score_head=nn.Linear(256, 80),
    box_head=nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4)),
)
```

The score head produces `[B,S,K]` logits and the box head produces `[B,S,4]`
coordinate deltas. Incompatible output shapes raise an error. This example is
a research architecture change, not a parity-preserving replacement by default.

## Tensor annotation conventions

Use [jaxtyping array annotations](https://docs.kidger.site/jaxtyping/api/array/)
with `torch.Tensor` at public boundaries:

- `Float[Tensor, "batch sequence channels"]` for projected memory.
- `Float[Tensor, "batch queries 4"]` for normalized box coordinates/logits.
- `Int64[Tensor, "batch queries"]` for selection indices.
- `Bool[Tensor, "1 sequence 1"]` for anchor validity (`True=valid`).

Use `Float`, not `Float32`, when a boundary accepts multiple floating precisions.
Geometry's `*shape` preserves arbitrary input rank; the box head's `*batch`
preserves all leading dimensions. Feature levels use `_channels _height _width`
because these axes can vary across levels; `FeatureSpec` and explicit validators
define their correspondence. Coordinate convention and detach policy still need
prose: shape annotations alone do not encode them.

This integration adds annotations without automatic runtime decorators. ty checks
the underlying Python types, while existing `validate()` methods enforce runtime
contracts. Matching symbolic axis names does not create runtime checks by itself.
The [upstream FAQ](https://docs.kidger.site/jaxtyping/faq/) explains static-checker
limits and Ruff's shape-string parsing. We exempt `F722` only in modules using
these strings. For a single named axis, the leading space in `" levels"` avoids
a false `F821` without disabling undefined-name checks.

## Gradient ownership

`QuerySet.ref_logits` still has gradients at this boundary. The future query
assembler must concatenate optional DN reference logits and detach the result
before decoding. Encoder auxiliary box predictions retain their own graph. The
tests explicitly distinguish these two routes.

This is a deliberately narrow first boundary. D-FINE distribution state, query
groups, AIFI feature taps, training methods, and capability validation are planned
in the [proposal](design-proposal.md). They are not simulated with placeholder APIs.
