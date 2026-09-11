# Deployment validation

`tools/deployment_validation.py` is a checkout-only probe for the official
R50vd EMA checkpoint at the fixed evaluation shape `(1, 3, 640, 640)`. It
records the checkpoint SHA-256, GPU and PyTorch/CUDA versions, warmup and timed
iteration counts, median/mean/p95 latency, and both peak allocated and reserved
CUDA memory. `model_only` times a device-resident tensor;
`end_to_end` uses the public Pillow-backed preprocessing and postprocessing
interfaces and includes CPU resize/normalization, host-to-device transfer, model
execution, and top-k box conversion. Model-only time uses CUDA events;
end-to-end time uses a synchronized host wall clock. The end-to-end input is an
already decoded CPU uint8 tensor and the output remains on CUDA. Image decoding
and device-to-host output transfer are not measured.

Run it on the target GPU after installing the core package and checkout
compatibility helpers:

```sh
PYTHONPATH=packages/core/src \
python tools/deployment_validation.py \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --output outputs/t4-deployment.json
```

Pass `--onnx path/to/model.onnx` and optionally `--onnx-fp16 path/to/model-fp16.onnx`
to attempt CUDA and TensorRT execution. The runtime profile counts node kernel
events by provider, so CPU fallback is based on executed nodes rather than the
provider list. TensorRT is successful only when its profile contains TensorRT
kernel events. Unavailable providers, preparation failures, and engine build
failures are retained as explicit results.

The graph exposes logits and boxes but not selector source indices. The report
therefore calls an in-order numerical comparison `ordered_rows_close` and marks
query identity alignment `unverified`; it does not infer query identity from row
position.

`tools/export_fp16_experiment.py` attempts a native CUDA FP16 graph through a
checkout-only adapter subclass and records unsupported failures. This experiment
does not expand the public export adapter's CPU FP32 contract.

## T4 result

The bounded official-EMA run used a Tesla T4, PyTorch 2.13.0+cu126, CUDA 12.6,
ONNX Runtime 1.23.2, ten warmups, and 50 timed iterations. FP32 model-only
latency was 51.63 ms median (54.64 ms p95) with 336,483,840 peak allocated
bytes. Its full pipeline measured 96.46 ms median. Native FP16 executed with
finite outputs at 32.01 ms model-only median and 82.07 ms full-pipeline median;
its ordered rows did not match FP32, so this is execution and performance
evidence rather than validated prediction parity. Reserved-memory values in the
report describe the warmed allocator and are not a minimum VRAM requirement.

The FP32 ONNX graph executed through the CUDA provider with finite, ordered-close
outputs (maximum differences 0.0003171 logits and 0.00001535 boxes). Profiling
recorded 1,573 CUDA and 20 CPU node-kernel events, so this is CUDA execution with
measured CPU fallback. The experimental FP16 graph also executed through CUDA,
but it failed ordered parity against native FP16 and remains unsupported.
Selector source indices are absent from both graphs, leaving query identity
alignment unverified.

The public FP32 export gate failed its seeded CPU runtime parity check even
though the adapter exactly matched eager. The artifact was retained only for
deployment diagnosis. TensorRT was not validated: direct attempts either exited
without a confirmed cause or reached the 300-second bound. A checkout-only copy
optimization removed all 20 `SplitToSequence` and 60 `SequenceAt` nodes and
passed the ONNX checker, but changed FP32 CPU outputs beyond tolerance; FP16
preparation failed ORT type validation. TensorRT inference was therefore not run
on either prepared graph. See `benchmarks/t4-deployment.json` for exact
provenance, measurements, provider counts, and bounded failure results.
