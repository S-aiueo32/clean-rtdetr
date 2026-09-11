# GPU execution and validation

The built-in detector supports FP16 weights and CUDA FP16 autocast. Position
embeddings retain upstream FP32 arithmetic under autocast and follow feature
precision for explicitly converted model weights. Deformable interpolation and
iterative box coordinates use FP32 when features are FP16/BF16. Public logits and boxes
remain FP32 (FP64 execution remains FP64). No caller-side cast is needed before
`SetCriterion` or denoising losses. This does not promise identical selected
queries or predictions across precisions: small score changes can change top-k
selection. Evaluation computes anchor logits on CPU before moving them to the
feature device, matching upstream fixed-size inference and the export path.
Training retains device-local anchor logarithms, matching upstream dynamic anchors.

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

## T4 precision regression recheck

The [T4 recheck report](../benchmarks/t4-precision-recheck.json) records 48
passing CPU/CUDA stage, precision, and CUDA AMP regression cases, with no skips on T4.
The default scaler (65536) took eleven attempts to make three parameter updates
and ended at scale 256. Starting at 128 made three updates in three attempts.
Skipped attempts preserved optimizer parameters; every update had finite,
nonzero gradients and changed parameters.

Native half and AMP also produced finite FP32 predictions for the full R50vd
preset with official EMA weights and a synthetic 640x640 input. This is an
execution result, not cross-precision numerical parity. With the recorded input,
AMP selected the same 300 query IDs as FP32 in a different order, but aligned
logits and boxes still exceeded the exploratory atol=0.005, rtol=0.05 comparison.
Native half shared 299 of 300 query IDs. Random-weight AMP shared 278 of 300
and native half shared 280 of 300; their common-subset comparisons passed while
full-query parity remained false.
The report preserves both execution and numerical outcomes separately.

An independent, pinned upstream comparison uses the same official EMA weights,
seed 41, and synthetic 640x640 input. After retaining FP32 AIFI positions under
autocast and matching CPU evaluation anchor logarithms, FP32 and AMP each match
upstream exactly: all backbone/encoder levels, query order, final logits, and
boxes have zero measured difference. The FP32-to-AMP difference within upstream
is identical to the difference within Clean RT-DETR. This single-input parity
check does not establish mixed-precision dataset accuracy.

The recheck and upstream jobs use the same source archive, recorded by SHA256
in the report. After GPU execution, a test's boolean argument was made
keyword-only for lint compliance; model code and test assertions are unchanged.
Local validation passed 482 tests with 19 skips, plus static
checks, all three package builds, and publish dry-runs.

## Validation scope

Regression tests cover prediction dtypes, finite gradients, actual denoising
parameter updates, unequal/empty-target distributed loss normalization, and
Lightning checkpoint resume. GPU tests require CUDA and otherwise skip. Run:

```sh
uv run pytest packages/core/tests/test_precision.py \
  packages/core/tests/test_encoder.py packages/core/tests/test_stages.py \
  packages/training/tests/test_gpu_precision.py \
  packages/training/tests/test_cuda_amp.py \
  packages/training/tests/test_gpu_lightning.py \
  packages/training/tests/test_distributed_losses.py
```

On macOS, set `GLOO_SOCKET_IFNAME=lo0` when running the two-rank Gloo test;
its worker processes require permission to bind local sockets.

The CUDA scaler regression starts at both the default scale 65536 and scale 128.
It requires three actual parameter updates within 24 attempts. Overflow attempts
must reduce the scale and leave optimizer parameters unchanged; every executed
update must have finite, nonzero gradients and change parameters. A three-batch
probe can end during initial scale calibration and cannot establish that training
never updates. Use the scaler's skip information when interpreting short runs.

For precision comparisons, record the selector's `source_indices` along with
predictions. Compare rows by query ID when both runs select the same set. If the
sets differ, report their overlap and any common-subset differences separately;
common-subset agreement does not establish full-output parity. Preserve the
original ordered comparison as a diagnostic. Keep numerical comparison outcomes
separate from finite-output execution checks, and do not loosen tolerances to
hide selection changes. Official-weight AMP accuracy requires dataset evaluation.

CPU BF16 autocast is covered for core inference and native DN training. CUDA
BF16 has not been validated: the T4 reports
`torch.cuda.is_bf16_supported(including_emulation=False) == False`. The Lightning
adapter does not accept `bf16-mixed`.

These checks use synthetic inputs and short training runs. They do not validate
full training convergence, mixed-precision COCO AP, parity across arbitrary
inputs or precisions, latency, or TensorRT.
