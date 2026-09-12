# Inference I/O

English | [日本語](ja/inference.md)

Use this guide to convert the supported official R50vd inference checkpoint,
run image inference, interpret its tensors, or export the validated ONNX
profile. Its examples use CPU FP32 R50vd. See [GPU execution and
validation](gpu-validation.md) for the bounded CUDA and AMP scope. The tools do
not download images or weights.

## Install and prepare a checkpoint

Python 3.13 or newer and PyTorch are required. Install Pillow before using image
files; install the separate export package only for ONNX:

```sh
python -m pip install "clean-rtdetr[inference]"
python -m pip install clean-rtdetr-export  # optional: ONNX export
```

In a source checkout, use `uv sync --package clean-rtdetr --extra inference`.
For export, use `uv sync --package clean-rtdetr-export`. Prefix Python and
package CLI commands below with `uv run`; do not prefix `curl`.

The packaged converter accepts the pinned official R50vd envelope and writes a
native plain `state_dict`. Download the envelope yourself, then convert it:

```sh
curl --fail --location --create-dirs \
  --output official.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

The default source is the pinned EMA weights and the converter verifies the
pinned SHA-256. Its JSON output records the native file hash and official load
report. Use `--source` and `--expected-sha256` only when converting a matching
official checkpoint with different validation requirements. Native loaders
accept an optional `expected_sha256`; prediction itself has no hash option.

<a id="command-line-example"></a>

## Predict an image

With `image.jpg` and `native.pth` available locally, run:

```sh
clean-rtdetr-predict image.jpg \
  --checkpoint native.pth --conf-thresh 0.5 --visualize-dir outputs
```

The command accepts one or more image paths, runs all six decoder layers, and
writes one JSON record per input to stdout. `--device` defaults to `cpu`.
It selects the top 300 query/class pairs, then `--conf-thresh` retains scores
greater than or equal to its finite `[0, 1]` value (default `0`). An input with
no retained candidates has empty arrays.

`--visualize-dir` writes PNGs such as `outputs/0001-image.png`. They use the
same candidates as stdout, show COCO names and scores, and retain source image
dimensions. The command creates the directory, but rejects a destination that
matches an input image; reruns replace visualization files. Drawing clips boxes
to the canvas, while JSON preserves the postprocessed coordinates.

<a id="python-api"></a>

## Use the Python API

`preprocess_images` needs Pillow and accepts a nonempty sequence of Pillow
images or CPU `uint8` RGB tensors shaped `[3,H,W]`. It converts every input
through Pillow RGB conversion and bilinear resizing to 640×640, returning CPU
FP32 values in `[0,1]`. It does not letterbox, apply mean/std normalization, or
apply EXIF rotation.

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images

model = rtdetr_r50vd().eval()
load_model_weights("native.pth", model=model)

with Image.open("image.jpg") as image:
    batch = preprocess_images([image])
with torch.inference_mode():
    output = model(batch.images, plan=DecoderPlan(collect_intermediate=False))
detections = postprocess(output.final, batch.original_sizes_wh)
```

Move `batch.images` to the model device before calling the model. The returned
`PreprocessedImages.images` is `[B,3,640,640]`; `original_sizes_wh` is CPU
int64 `[B,2]` in `(width, height)` order. `postprocess` moves those sizes to the
prediction device when needed.

<a id="model-tensors"></a>

## Model and postprocessing contracts

`RTDETR` accepts preprocessed floating-point RGB `[B,3,H,W]` tensors. Height
and width must be multiples of 32, and the projected memory needs at least the
configured number of query positions. The model does no resizing or
normalization.

`ModelOutput.final` holds raw class logits `[B,Q,K]` and normalized `cxcywh`
boxes `[B,Q,4]` from the last executed decoder layer. `DecoderPlan()` executes
and collects all six layers by default; use
`DecoderPlan(collect_intermediate=False)` when only the final output is needed.

<a id="postprocessing"></a>

`postprocess(predictions, original_sizes_wh, top_k=300)` returns zero-based
`labels[B,T]`, sigmoid `scores[B,T]`, and original-image-pixel
`boxes_xyxy[B,T,4]`. It ranks every query/class pair, so one query can yield
several labels. It deliberately applies no NMS, score threshold, or clipping;
labels are model class indices, not COCO category IDs. `top_k` must be an
integer from 1 through `Q*K`. PyTorch determines tie ordering.

<a id="fixed-shape-onnx-export"></a>

## Export fixed-shape ONNX

`clean_rtdetr_export.export_onnx(model, destination, *,
input_shape=(1, 3, 640, 640))` writes a tensor-only `FixedShapeRTDETR` graph.
The validated profile is batch 1, CPU FP32 RGB input `[1,3,640,640]`, opset 18,
raw logits `[1,300,80]`, and normalized `cxcywh` boxes `[1,300,4]`. The CLI
loads a local native checkpoint; `--checkpoint` and `--output` are required,
`--expected-sha256` is optional, and `--report` defaults to the output path with
a `.json` suffix:

```sh
clean-rtdetr-export --checkpoint native.pth --output outputs/rtdetr.onnx
```

Export verifies eager PyTorch and ONNX Runtime on seeded synthetic inputs 41 and
42 (`atol=1e-5`, `rtol=1e-4`). It uses `optimize=False` and disabled ONNX Runtime
graph optimization because that is the verified numerical profile. Run the
Python API example above first so that `batch.images` contains its preprocessed
image, then run the graph with the same settings:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "outputs/rtdetr.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)
# batch.images is CPU float32 RGB [1,3,640,640] in [0,1].
logits, boxes = session.run(["logits", "boxes"], {"images": batch.images.numpy()})
```

The wrapper copies the source model to CPU FP32 evaluation mode and leaves the
source model unchanged. It validates tensor shape, dtype, and evaluation mode.
Smaller fixed shapes support regression tests; the recorded R50vd profile is
640×640. Dynamic shapes, TensorRT, AP, and latency are outside this validation.

<a id="official-weights-and-parity"></a>

## Official-weight compatibility

`clean-rtdetr-convert` is a public command-line interface; its official-weight
conversion implementation is private and has no public compatibility API. In a
checkout, `tools.compatibility.load_official_model_weights` and
`tools.compatibility.load_official_stage_weights` load the original official
envelope with an explicit `model` or `ema` source. Packaged code instead loads
the native file through `clean_rtdetr.checkpoint.load_model_weights`.

The pinned URL, hash, and source are in
[`benchmarks/baseline.json`](../benchmarks/baseline.json). For a synthetic CPU
FP32 parity comparison, use `tools/checkpoint_parity.py`; `--full-model`
compares a seeded 640×640 input and does not measure dataset accuracy. Its
recorded full-model result is
[`benchmarks/full-model-parity.json`](../benchmarks/full-model-parity.json).
