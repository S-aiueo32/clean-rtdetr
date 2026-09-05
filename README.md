# Clean RT-DETR

A readable RT-DETR reference implementation with composable research extensions.
The design separates model variants, training methods, and experiment recipes
from deployment transformations.

## Current implementation

The current implementation connects **images → R50vd backbone → HybridEncoder →
projected memory → selected queries → iterative decoder predictions** in PyTorch.

- Frozen feature/query contracts with jaxtyping shape/dtype annotations and eager validation.
- Decoder input projection, row-major flattening, and explicit level offsets.
- Anchor generation and top-k selection preserving upstream masking semantics.
- Feature-content detach and differentiable encoder auxiliary predictions.
- DN-ready query assembly with an explicit reference-logit detach boundary.
- Self-attention, multi-scale deformable cross-attention, and iterative box refinement.
- Explicit decoder execution/output plans instead of implicit inference-layer selection.
- Forward and gradient parity tests against SHA-256-pinned upstream methods.
- An explicit R50vd factory and component-injected `RTDETR`, with optional AIFI features.
- Strict official checkpoint loading for all inference stages, with a reproducible parity example.

Preprocessing, postprocessing, denoising construction, training criteria, COCO
evaluation, and export remain to be implemented. No AP or latency is claimed.
`examples/query_selection.py` continues to demonstrate the component path using
synthetic features and randomly initialized weights.

## Image-to-prediction API

```python
import torch
from clean_rtdetr import DecoderPlan, build_rtdetr_r50vd

model = build_rtdetr_r50vd().eval()  # Random weights; no automatic download.
with torch.inference_mode():
    output = model(
        torch.rand(1, 3, 640, 640),
        plan=DecoderPlan(collect_intermediate=False),
        collect_aifi=True,
    )
print(output.final.logits.shape)  # [1, 300, 80], raw class logits
print(output.final.boxes.shape)  # [1, 300, 4], normalized cxcywh
```

Inputs are preprocessed floating-point RGB tensors in `[B, 3, H, W]` layout.
Height and width must be multiples of 32, and the resulting memory must have at
least 300 positions for the default query count. The model does not resize or
normalize images. The output always contains `decoder`, differentiable
`encoder_aux`, and optional `after_aifi` fields; `final` selects the last executed
decoder layer. `DecoderPlan()` collects all six layers by default in both modes.

Use `clean_rtdetr.compatibility.load_official_model_weights` with explicit
`model`, `source`, and `expected_sha256` arguments to load an official artifact.
The existing `load_official_stage_weights` API remains available for the three
decoder-side components. Both loaders validate all weights before changing any
component and report key mappings, exclusions, and synthesized BatchNorm counters.

To compare every inference stage against the pinned upstream implementation,
obtain the artifact identified in `benchmarks/baseline.json`, then run:

```sh
uv run python examples/checkpoint_parity.py \
  checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --full-model --report outputs/full-model-parity.json
```

Create the report's parent directory before running. This compares a seeded
synthetic 640×640 input in FP32 eager execution; it does not measure detection
accuracy on a dataset. Omitting `--full-model` retains the synthetic-feature
stage comparison. Checkpoints and generated reports are local artifacts.
The [recorded CPU FP32 result](benchmarks/full-model-parity.json) matches every
compared stage exactly for this synthetic input and includes source hashes for
the implementation used in the run.

## Contributing

Development setup, validation commands, linting policy, and implementation notes
are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

For a quick local check, install the locked development environment and run:

```sh
uv sync --locked
uv run pytest
```

## Research direction

The [design proposal](docs/design-proposal.md) describes the planned
`ModelVariant × TrainingMethod × ExperimentRecipe` composition. RT-DETRv3's
auxiliary branches, RT-DETRv4-style distillation, DEIM recipes, and D-FINE-style
distribution refinement inform these boundaries. They are design targets, not
implemented integrations. New regression representations may replace an entire
decoder/selector boundary rather than conforming to RT-DETR coordinate deltas.

See the [implementation roadmap](docs/roadmap.md) and the
[extension example](docs/extensions.md) for the next concrete steps.

## License

Clean RT-DETR is licensed under the [Apache License, Version 2.0](LICENSE).
The vendored RT-DETR source and adaptations retain the upstream Apache-2.0
license and attribution notices described in [third_party/RT-DETR/README.md](third_party/RT-DETR/README.md).

## Provenance

Based on [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), pinned to commit
`068dfde65f2667ad6555883c69d73de886518cad`. Upstream-derived modules carry change
notices. The [oracle source and license](third_party/RT-DETR/README.md) are retained
for parity testing. [Baseline metadata](benchmarks/baseline.json) explicitly marks
checkpoint, AP, and latency measurements as pending.
