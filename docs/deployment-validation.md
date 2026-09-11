# Deployment validation

`tools/deployment_validation.py` is a checkout-only probe for the official
R50vd EMA checkpoint at the fixed evaluation shape `(1, 3, 640, 640)`. It
records the checkpoint SHA-256, GPU and PyTorch/CUDA versions, warmup and timed
iteration counts, median/mean/p95 latency, and both peak allocated and reserved
CUDA memory. `model_only` times a device-resident tensor;
`end_to_end` uses the public Pillow-backed preprocessing and postprocessing
interfaces and includes CPU resize/normalization, host-to-device transfer, model
execution, and top-k box conversion. Model-only time uses CUDA events;
end-to-end time uses a synchronized host wall clock.

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
provider list. A TensorRT provider session creates an engine and runs inference;
unavailable providers and engine build failures are retained as explicit results.

The graph exposes logits and boxes but not selector source indices. The report
therefore calls an in-order numerical comparison `ordered_rows_close` and marks
query identity alignment `unverified`; it does not infer query identity from row
position.

`tools/export_fp16_experiment.py` attempts a native CUDA FP16 graph through a
checkout-only adapter subclass and records unsupported failures. This experiment
does not expand the public export adapter's CPU FP32 contract.
