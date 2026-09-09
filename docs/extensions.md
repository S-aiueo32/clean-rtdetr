# Extension boundaries

English | [日本語](ja/extensions.md)

## Named model presets

`create_model(name, num_classes=80)` constructs the official PResNet variants
listed by `list_models()`: R18vd, R34vd, R50vd-M, R50vd, and R101vd for both
RT-DETR and RT-DETRv2. The corresponding named functions remain available when
a statically precise return type is useful.

```python
from clean_rtdetr import create_model, get_model_preset

model = create_model("rtdetr_r101vd", num_classes=20)
preset = get_model_preset("rtdetr_r101vd")
print(preset.config_url, preset.checkpoint_url)
```

`ModelPreset` freezes the resolved architecture values and upstream provenance.
The URLs do not trigger downloads. Only the existing `rtdetr_r50vd` official
conversion and parity path is validated; the other constructors currently cover
architecture and state-dictionary structure.

The v2 presets inject named encoder projections, a named selector projection,
point-scale state, and identity FFN residual handling into the shared stages.
The medium presets also inject a default inference `DecoderPlan`; training and
an explicit depth still execute the requested decoder prefix. Model execution
contains no v1/v2 family switch.

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

## Shared contracts and geometry

Shared tensor payloads and their validation live in `clean_rtdetr.contracts`.
Anchor generation and coordinate transforms live in `clean_rtdetr.geometry`.
Model stage implementations and interfaces live in `clean_rtdetr.models`.

```python
from clean_rtdetr.contracts import DecoderPlan, EncodedFeatures
from clean_rtdetr.geometry import generate_anchors, inverse_sigmoid
```

Imports from `clean_rtdetr.core.contracts` and `clean_rtdetr.core.geometry`
must use these paths; the `clean_rtdetr.core` subpackage has been removed.
Existing package-root imports such as `from clean_rtdetr import DecoderPlan`
remain available. Tensor state-dict keys and computation are unchanged.

## Type parameters and aliases

Use [PEP 695](https://peps.python.org/pep-0695/) syntax for generic classes,
functions, and type aliases. The workspace requires Python 3.13 or newer, and
Ruff targets Python 3.13 to enforce this convention. Declare parameters in their
own scope (`class TrainingModel[PredictionT]`), retain component bounds
(`DetectorT: Detector`), and let the type checker infer variance.

Declare aliases with `type`, including jaxtyping tensor contracts.
`BaselineRTDETR` is an annotation alias; construct models with `RTDETR(...)` or
`rtdetr_r50vd()`. Type aliases are evaluated lazily and are not runtime classes
or constructors. `ImageInput` keeps its static Pillow/tensor union and a runtime
`object` alias so Pillow remains optional and explicit input validation retains
its existing errors.

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

`QuerySet.ref_logits` still has gradients at this boundary. `QueryAssembler`
concatenates optional DN reference logits and detaches the result
before decoding. Encoder auxiliary box predictions retain their own graph. The
tests explicitly distinguish these two routes.

The four independent recipes now exercise custom classification, distribution
decoding, query branches and AIFI distillation through public interfaces.
See [recipe responsibilities](recipe-boundaries.md) for the resulting library
APIs and the algorithms, state and validation that remain user-owned. The
[proposal](design-proposal.md) records broader design ideas; it is not a support
matrix for those algorithms.
