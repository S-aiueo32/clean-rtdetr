# Clean RT-DETR

A readable RT-DETR reference implementation with composable research extensions.
The design separates model variants, training methods, and experiment recipes
from deployment transformations.

## Current implementation

The current slice implements **RGB images → R50vd backbone → HybridEncoder →
projected memory → selected queries → iterative decoder predictions** in PyTorch.

- R50vd with official stem/stage freezing and frozen normalization semantics.
- HybridEncoder with an optional differentiable AIFI feature boundary before cross-scale fusion.
- Frozen feature/query contracts with jaxtyping shape/dtype annotations and eager validation.
- Decoder input projection, row-major flattening, and explicit level offsets.
- Anchor generation and top-k selection preserving upstream masking semantics.
- Feature-content detach and differentiable encoder auxiliary predictions.
- DN-ready query assembly with an explicit reference-logit detach boundary.
- Self-attention, multi-scale deformable cross-attention, and iterative box refinement.
- Explicit decoder execution/output plans instead of implicit inference-layer selection.
- Forward and gradient parity tests against SHA-256-pinned upstream methods.
- Strict official checkpoint loading for all inference stages, preserving the stage-only loader.

`build_rtdetr_r50vd()` builds the R50vd backbone, HybridEncoder, and decoder pipeline.
`load_official_model_weights` in `clean_rtdetr.compatibility` loads official weights.
Model inputs are floating point RGB
`[0, 1]` NCHW tensors with height and width divisible by 32; inputs must provide at
least as many feature tokens as configured queries. Prepare RGB float values in `[0, 1]`
**without mean/std normalization**. The model does not implement preprocessing,
postprocessing, denoising construction, training criteria, COCO AP evaluation,
latency benchmarking, or export. CPU FP32 stage parity is validated at 640×640;
this is not yet a complete evaluated detector.

```python
import torch
from clean_rtdetr import build_rtdetr_r50vd

model = build_rtdetr_r50vd().eval()  # random weights; no downloads
with torch.inference_mode():
    output = model(torch.rand(1, 3, 640, 640), collect_aifi=True)
print(output.final.logits.shape)  # [1, 300, 80]
print(output.final.boxes.shape)  # [1, 300, 4], normalized cxcywh
```

For checkpoint parity, use the reproducible module example (with a local checkpoint):

```sh
uv run python -m examples.model_checkpoint_parity CHECKPOINT \
  --source ema --expected-sha256 SHA256 --report REPORT.json
```

The existing stage-only check remains available as
`uv run python -m examples.checkpoint_parity`.

The checked-in [stage-parity report](benchmarks/r50vd-stage-parity.json) records
the seed, environment, source hashes, and numerical comparisons. Its base Git
revision and source hashes together identify the validated implementation; the
example also emits the complete checkpoint key mapping when rerun.

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
for parity testing. [Baseline metadata](benchmarks/baseline.json) records the official
EMA artifact and the CPU stage-parity report; COCO AP and GPU latency remain pending.
