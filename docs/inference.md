# Inference I/O

English | [日本語](ja/inference.md)

Pillow is an optional dependency. Install image support with:

```sh
uv sync --package clean-rtdetr --extra inference
```

For ONNX export and its CPU ONNX Runtime verifier, install the separate
`clean-rtdetr-export` workspace package with
`uv sync --package clean-rtdetr-export`.

## Model tensors

`RTDETR` accepts preprocessed floating-point RGB tensors in `[B,3,H,W]` layout.
Height and width must be multiples of 32, and the resulting memory must contain
at least the configured number of query positions. The model does not resize or
normalize inputs.

`ModelOutput.final` selects the last executed decoder layer and contains raw
class logits `[B,Q,K]` and normalized cxcywh boxes `[B,Q,4]`. The full output
also carries differentiable encoder predictions and optional AIFI features.
`DecoderPlan()` executes and collects all six layers by default.

## Python API

After installing image support, provide `image.jpg` and the local official
checkpoint listed in [baseline metadata](../benchmarks/baseline.json):

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images

model = rtdetr_r50vd().eval()
load_official_model_weights(
    "checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth",
    model=model,
    source="ema",
    expected_sha256="dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45",
)
with Image.open("image.jpg") as image:
    batch = preprocess_images([image])
with torch.inference_mode():
    output = model(batch.images, plan=DecoderPlan(collect_intermediate=False))
detections = postprocess(output.final, batch.original_sizes_wh)
```

`preprocess_images` accepts a nonempty sequence of Pillow images and/or CPU RGB
`uint8 [3,H,W]` tensors, including images of different sizes. Both paths use
Pillow RGB conversion and bilinear resize to 640×640, then convert to CPU FP32
`[0,1]`. There is no letterbox, mean/std normalization, or EXIF rotation.

`PreprocessedImages.images` has shape `[B,3,640,640]`.
`original_sizes_wh` is CPU int64 `[B,2]` in W,H order. Move `images` to the model
device explicitly; postprocessing transfers the size tensor as needed.

## Postprocessing

`postprocess(predictions, original_sizes_wh, top_k=300)` returns zero-based
`labels[B,T]`, sigmoid `scores[B,T]`, and original-image pixel
`boxes_xyxy[B,T,4]`. It ranks all query/class pairs, so one query may produce
multiple labels. Tie ordering follows PyTorch.

The upstream-compatible path applies no NMS, score threshold, or box clipping.
Labels are model class indices rather than COCO category IDs. `top_k` must be
between 1 and the number of query/class pairs.

## Command-line example

`clean-rtdetr-predict` accepts one or more image paths, a local official
checkpoint, its selected source, and expected SHA-256. It prints all 300
detections per image as JSON and runs all six decoder layers by default. The
example does not download images or weights; `--device` defaults to `cpu`.

```sh
uv run clean-rtdetr-predict image.jpg \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45
```

## Fixed-shape ONNX export

`clean_rtdetr_export` exposes the `FixedShapeRTDETR` tensor-only wrapper and
the `export_onnx(model, destination, *, input_shape=(1, 3, 640, 640))` API.
The default profile is batch 1, 640×640, CPU FP32, and ONNX opset 18. It emits
raw class logits `[1,300,80]` and normalized cxcywh boxes `[1,300,4]`, then
checks ONNX Runtime parity against eager PyTorch for seeds 41 and 42 with
`atol=1e-5` and `rtol=1e-4`. The tested profile uses `optimize=False` during
export and disables ONNX Runtime graph optimizations. Default automatic
optimizations failed the numerical gate on the same inputs; this profile does
not validate optimized execution.

The wrapper owns an eval CPU FP32 copy and preserves the source model's weights
and training state. `Float32` jaxtyping annotations describe its input and output
tensors; explicit checks also reject wrong shapes, dtypes, and training mode.
Lowering requires the built-in selector and decoder types, so overriding their
forward methods cannot silently change the exported behavior. Smaller fixed
shapes are accepted for regression tests; the recorded R50vd profile is 640×640.

To run the artifact with the validated settings:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "outputs/rtdetr.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)
# images: CPU float32 [1,3,640,640], RGB in [0,1].
logits, boxes = session.run(["logits", "boxes"], {"images": images.numpy()})
```

Run the CLI with
`uv run clean-rtdetr-export`;
`--checkpoint`, `--source`,
`--expected-sha256`, and `--output` are required. `--report` optionally selects
the JSON report path (default: output with a `.json` suffix). It loads only a
local, SHA-256-verified official checkpoint and does not download weights.
Variable/dynamic shapes, TensorRT, AP, and latency are out of scope.

## Official weights and parity

`load_official_model_weights` requires the checkpoint path, its `model` or `ema`
source, and expected SHA-256. It validates every tensor before changing the
model. `load_official_stage_weights` remains available for decoder-side
components.

The checkpoint URL, hash, and source are recorded in
`benchmarks/baseline.json`. Compare all implemented inference stages with:

```sh
uv run python tools/checkpoint_parity.py \
  checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --full-model --report outputs/full-model-parity.json
```

Create the report's parent directory first. The command compares a seeded
synthetic 640×640 input in CPU FP32; it does not measure dataset accuracy.
Omitting `--full-model` runs the synthetic-feature stage comparison. The
recorded result is in `benchmarks/full-model-parity.json`.
