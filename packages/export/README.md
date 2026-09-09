# Clean RT-DETR Export

[日本語版](README.ja.md)

`clean-rtdetr-export` exports the R50vd detector to a fixed-shape ONNX graph and
checks CPU FP32 ONNX Runtime parity with PyTorch. It requires Python 3.13 or
newer and depends on the matching `clean-rtdetr` release.

```sh
python -m pip install clean-rtdetr-export
```

The validated export profile is batch 1, RGB float32 input `[1, 3, 640, 640]`,
CPU execution, ONNX opset 18, and raw outputs of logits `[1, 300, 80]` and
normalized cxcywh boxes `[1, 300, 4]`. Exporter and runtime graph optimizations
are disabled in the validated profile. The export CLI requires a local converted
native state-dict checkpoint and optionally validates its SHA-256; it does not
download weights. Download and convert an official envelope with the
`clean-rtdetr-convert` command documented in the [inference guide](../../docs/inference.md).
Dynamic shapes, TensorRT, AP, and latency are outside the validated scope.

For development from source:

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-export
uv run clean-rtdetr-export --help
uv run pytest packages/export/tests
```

Export a checkpoint and write the verification report with:

```sh
uv run clean-rtdetr-export \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --output outputs/rtdetr.onnx
```

The Python API is exposed from `clean_rtdetr_export`. `export_onnx` accepts an
already constructed evaluation model; `FixedShapeRTDETR` is the tensor-only
wrapper used for lowering. This package does not install the training package
and does not provide a `clean_rtdetr.export` compatibility shim.

Clean RT-DETR is Apache-2.0 licensed and is based on `lyuwenyu/RT-DETR` at
commit `068dfde65f2667ad6555883c69d73de886518cad`; the workspace root retains
the upstream source, license, and detailed provenance.
