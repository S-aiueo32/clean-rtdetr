# Native BF16 validation

The earlier native L4 fixes are already represented by the current low-precision
sampling and FP32 prediction-boundary implementation. This integration extends
the precision test matrix to CUDA BF16, requires native hardware support, and
covers learned and selected query content. BF16 training uses FP32 parameters
without GradScaler; FP16 retains its scaler calibration checks.

An NVIDIA L4 in Lightning Studio passed all 13 selected precision regressions
with zero skips. This includes three actual compact DN optimizer updates under
native BF16. Official EMA full R50vd inference on a seed-41 synthetic 640×640
input also produced finite FP32 logits and boxes. The [record](../benchmarks/bf16-studio.json)
contains the source/input hashes and environment. The Studio was stopped after
validation; the dependency freeze and raw artifacts are retained locally under
`outputs/lightning/20260911-bf16/` in the original checkout.

## Reproduce

On a native BF16 GPU, install this checkout's core inference, training, and test
dependencies. Supply a converted native official EMA state dictionary:

```sh
uv sync --all-packages --all-extras
uv run python tools/bf16_validation.py \
  --checkpoint /data/official-ema.pt --expected-sha256 NATIVE_SHA256 \
  --output outputs/bf16-validation
```

The tool rejects CPU/emulated BF16, requires every selected regression to run,
and writes the JUnit report, logs and JSON result. It does not establish BF16
COCO AP, full training convergence, DDP/resume, Lightning `bf16-mixed`, or TensorRT
support. Ordered FP32/BF16 differences are descriptive because query selection
can change; they are not treated as aligned parity.
