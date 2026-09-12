# GPU execution and validation

English | [日本語](ja/gpu-validation.md)

This page defines the bounded CUDA and mixed-precision evidence for Clean
RT-DETR. Use it to choose a supported execution profile and to interpret the
recorded GPU checks. It is not a claim of training convergence, dataset AP,
latency, TensorRT support, or numerical equality across precisions.

## Precision behavior

The detector accepts FP16 weights and CUDA FP16 autocast. Position embeddings
retain upstream FP32 arithmetic under autocast; deformable interpolation and
iterative box coordinates use FP32 when features are FP16/BF16. Public logits
and boxes remain FP32 (and remain FP64 in FP64 execution), so `SetCriterion`
and DN losses need no caller-side cast.

Precision can change top-k scores and therefore selected queries. Treat finite
outputs as an execution check, not parity. For a numerical comparison, record
the selector's `source_indices`, compare rows by query ID when both runs select
the same IDs, and report overlap plus common-subset differences when they do
not. Common-subset agreement does not establish full-output parity.

For mixed-precision training, retain FP32 model weights and use autocast with
`torch.amp.GradScaler("cuda")`. An attempted batch may not update parameters
while the scaler backs off; an epoch scheduler still follows completed epochs.
Pure-FP16 inference does not validate pure-FP16 optimizer training.

## Lightning CUDA profile and DDP boundary

The shipped Lightning adapter supports one process: CPU `32-true`/`bf16-mixed`
and CUDA `32-true`/`16-mixed`. Select CUDA with:

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training trainer=cuda
```

The CUDA profile uses `deterministic: warn` because CUDA `grid_sample` backward
does not guarantee deterministic accumulation. Completed-epoch checkpoints retain
CUDA and CPU RNG, the DataLoader generator, optimizer, scheduler, AMP scaler,
and EMA. AMP resume is compared with tolerances, not bitwise identity. See
[Lightning training](lightning.md) for its exact-resume requirements.

Native PyTorch DDP loss calculation is a separate supported boundary: it reduces
the target count with `all_reduce(sum) / world_size`, clamps it to one, and uses
that global average as the normalizer. Every rank must call the criterion in the
same order. This validation does not extend the Lightning adapter to multiple
ranks, gradient accumulation, or worker-based exact resume.

## Recorded checks

The [Kaggle record](../benchmarks/gpu-validation.json) documents two Tesla T4
GPUs, Python 3.13.13, PyTorch 2.13.0+cu126, source identity, seeds, and results.
It records full random-weight R50vd FP16/autocast inference with finite FP32
outputs; compact native AMP training with DN and auxiliary losses; two-rank NCCL
FP32/AMP steps; unequal/empty-target global normalization; and Lightning
FP32/AMP epoch-resume checks. These are short synthetic executions.

The [T4 precision recheck](../benchmarks/t4-precision-recheck.json) records 48
passing CPU/CUDA precision and AMP regression cases. It distinguishes finite
execution from cross-precision numerical comparison. With official EMA weights
on one synthetic 640×640 input, Clean RT-DETR and pinned upstream matched exactly
within each FP32 or AMP run; that single-input result does not establish
mixed-precision dataset accuracy.

The T4 did not report native CUDA BF16 support. A separate
[historical L4 BF16 probe](../benchmarks/bf16-l4.md) established native CUDA BF16
execution for a small synthetic model on its recorded source snapshot. It does
not validate the later T4 implementation, and the Lightning adapter does not
accept CUDA `bf16-mixed`.

## Reproduce the focused checks

Run this from the repository root after preparing the complete workspace:

```sh
uv sync --all-packages
```

The command includes CPU and two-rank Gloo tests, which can run without CUDA.
Only CUDA-specific test cases skip when CUDA is unavailable. See [Lightning
training](lightning.md#install-and-run) for the training-recipe setup.

```sh
uv run pytest packages/core/tests/test_precision.py \
  packages/core/tests/test_encoder.py packages/core/tests/test_stages.py \
  packages/training/tests/test_gpu_precision.py \
  packages/training/tests/test_cuda_amp.py \
  packages/training/tests/test_gpu_lightning.py \
  packages/training/tests/test_distributed_losses.py
```

On macOS, set `GLOO_SOCKET_IFNAME=lo0` for the two-rank Gloo test because its
workers bind local sockets. The CUDA scaler regression requires three successful
parameter updates within 24 attempts at each recorded initial scale; skipped
attempts must reduce the scale without changing parameters. A short probe that
only calibrates the scaler cannot show that training updates.

These checks use synthetic data and bounded runs. They do not validate full
training reproduction, mixed-precision COCO AP, arbitrary-input precision
parity, latency, dynamic-shape ONNX, or TensorRT.
