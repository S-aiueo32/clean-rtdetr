# GPU execution and validation

The built-in detector supports FP16 weights and CUDA FP16 autocast. Position
embeddings follow feature precision; deformable interpolation and iterative
box coordinates use FP32 when features are FP16/BF16. Public logits and boxes
remain FP32 (FP64 execution remains FP64). No caller-side cast is needed before
`SetCriterion` or denoising losses. This does not promise identical selected
queries or predictions across precisions: small score changes can change top-k
selection.

For mixed-precision training, keep model weights in FP32 and use autocast with
`torch.amp.GradScaler("cuda")`. Pure FP16 model inference does not establish
support for pure FP16 optimizer training. The scaler may skip initial updates
while reducing its scale; an attempted batch is not necessarily an optimizer
step. The epoch scheduler still follows completed epochs.

The Lightning recipe accepts single-process CPU `32-true` and CUDA `32-true`
or `16-mixed`. Select the CUDA profile with:

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training trainer=cuda
```

Supply the dataset and other recipe overrides described in [Lightning
training](lightning.md). The CUDA profile uses `deterministic: warn` because
CUDA `grid_sample` backward does not guarantee deterministic accumulation.
Epoch-boundary checkpoints retain the active CUDA device RNG, CPU RNGs,
DataLoader generator, optimizer, scheduler, scaler, and EMA state. EMA advances
only when the optimizer actually steps; AMP overflow skips do not update EMA.
CUDA resume is checked with numerical tolerances; bitwise reproducibility is
not promised. Legacy CPU epoch checkpoints remain accepted; CUDA resume
requires CUDA RNG state and matching execution settings.

Native PyTorch DDP losses normalize by the global average target count
(`all_reduce(sum) / world_size`, clamped to one), matching DDP's gradient
averaging. All ranks must participate in the same criterion calls. The
Lightning adapter still rejects multiple ranks, gradient accumulation, and
worker-based exact resume; native DDP validation does not extend that adapter's
checkpoint contract.

## Measured GPU checks

The [recorded Kaggle run](../benchmarks/gpu-validation.json) uses two Tesla T4
GPUs, Python 3.13.13, and PyTorch 2.13.0+cu126. This CUDA profile is separate
from the macOS development lock. The record includes the base commit, exact
source archive hash, seeds, and results; the job and source archive are retained
under `outputs/kaggle/20260911-fixes/` in the validation checkout.

| Check | Scope |
| --- | --- |
| FP16 weights / FP16 autocast | Full random-weight R50vd inference at 256×256; finite FP32 public outputs |
| Native AMP training | Compact model, DN and auxiliary losses, scaler backoff and three successful parameter updates |
| Native DDP FP32 / AMP | Two NCCL ranks, two optimizer steps, finite gradients and identical parameters across ranks |
| Global loss normalization | Two Gloo ranks with unequal/empty targets; losses and gradients match the concatenated objective |
| Lightning FP32 / AMP resume | Exact restored model, optimizer, EMA, CUDA RNG and scaler state; continued training compared with tolerances |

Resume comparisons use math SDPA, `deterministic="warn"`, and absolute/relative
tolerances of 1e-4 for FP32 and 2e-3 for AMP. CUDA interpolation backward remains
nondeterministic; these tolerances bound subsequent training drift, while the
state restored before the next step must match exactly. FP32 trains two epochs;
AMP trains twelve to include actual updates after scaler backoff. Epoch counts
are not optimizer update counts. See the record for measured EMA updates and
parameter differences.

## Validation scope

Regression tests cover prediction dtypes, finite gradients, actual denoising
parameter updates, unequal/empty-target distributed loss normalization, and
Lightning checkpoint resume. GPU tests require CUDA and otherwise skip. Run:

```sh
uv run pytest packages/core/tests/test_precision.py \
  packages/training/tests/test_gpu_precision.py \
  packages/training/tests/test_gpu_lightning.py \
  packages/training/tests/test_distributed_losses.py
```

CPU BF16 autocast is covered for core inference and native DN training. CUDA
BF16 has not been validated: the T4 reports
`torch.cuda.is_bf16_supported(including_emulation=False) == False`. The Lightning
adapter does not accept `bf16-mixed`.

These are short synthetic, random-weight checks. They do not validate full
training convergence, COCO AP, official-weight mixed-precision parity, latency,
or TensorRT.
