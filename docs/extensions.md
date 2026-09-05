# First extension boundary

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

`QuerySet.ref_logits` still has gradients at this boundary. The future query
assembler must concatenate optional DN reference logits and detach the result
before decoding. Encoder auxiliary box predictions retain their own graph. The
tests explicitly distinguish these two routes.

This is a deliberately narrow first boundary. D-FINE distribution state, query
groups, AIFI feature taps, training methods, and capability validation are planned
in the [proposal](design-proposal.md). They are not simulated with placeholder APIs.
