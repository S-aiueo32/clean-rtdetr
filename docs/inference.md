# Inference I/O

English | [日本語](ja/inference.md)

The core package includes checkpoint conversion. Install it with:

```sh
python -m pip install clean-rtdetr
```

Pillow is an optional dependency. Install image support with:

```sh
python -m pip install "clean-rtdetr[inference]"
```

In a checkout development environment, use `uv sync --package clean-rtdetr
--extra inference` and prefix the commands below with `uv run`.

For ONNX export and its CPU ONNX Runtime verifier, install the separate
`clean-rtdetr-export` package with `python -m pip install clean-rtdetr-export`.
In a checkout development environment, use `uv sync --package
clean-rtdetr-export`.

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

After installing image support, provide `image.jpg` and a converted native
state-dict checkpoint. From any directory, download the pinned official
checkpoint and convert it with the installed command:

```sh
curl --fail --location --create-dirs \
  --output official.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

With the checkout development environment, use `uv run clean-rtdetr-convert`
in place of the installed command.

The converter prints JSON containing the converted output path, its SHA-256,
and the official load report. It defaults to the pinned EMA source and checks
the pinned official SHA-256; `--source` and `--expected-sha256` remain available
when validating another matching checkpoint. The packaged loader accepts an
optional `expected_sha256` keyword for validating the converted file.
The image prediction CLI loads the converted checkpoint without a hash option.

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images

model = rtdetr_r50vd().eval()
load_model_weights(
    "native.pth",
    model=model,
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

`clean-rtdetr-predict` accepts one or more image paths and a local native
state-dict checkpoint. It prints detections per image as JSON to stdout and runs
all six decoder layers. By default, it retains all 300 candidates. The
example does not download images or weights; `--device` defaults to `cpu`.

```sh
clean-rtdetr-predict image.jpg \
  --checkpoint native.pth --conf-thresh 0.5 --visualize-dir outputs
```

`--conf-thresh` keeps candidates with a score greater than or equal to the given
threshold. It accepts finite values in `[0, 1]` and defaults to `0`. Filtering
applies after top-k selection and preserves score order. If no candidates pass,
the image record contains empty `labels`, `scores`, and `boxes_xyxy` arrays.

`--visualize-dir` optionally saves PNGs with boxes, COCO class names, and scores
for the same candidates printed to stdout. It creates the directory as needed
and preserves the original image dimensions. Files use a one-based input index
and image stem: `outputs/0001-image.png`, `outputs/0002-image.png`, and so on.
Existing visualization files are replaced on reruns; a destination matching any
input image is rejected before inference. With no candidates, it saves an
unannotated image. Drawing clips boxes to the canvas; JSON coordinates retain
the original postprocessing values. These CLI options do not change the Python
`postprocess` API or the COCO evaluation path.

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

Run the installed CLI with `clean-rtdetr-export`;
`--checkpoint` and `--output` are required. `--expected-sha256` optionally
validates the native checkpoint; `--report` selects the JSON report path
(default: output with a `.json` suffix). It loads only a local native
checkpoint and does not download weights.
Variable/dynamic shapes, TensorRT, AP, and latency are out of scope.

## Official weights and parity

The converter is an internal core CLI implementation and has no public
compatibility API. Official compatibility loaders are checkout-only: use
`tools.compatibility.load_official_model_weights` and
`tools.compatibility.load_official_stage_weights` with the original official
envelope and an explicit `model` or `ema` source. The packaged
`clean_rtdetr.checkpoint.load_model_weights(path, model=model,
expected_sha256=...)` accepts the native plain `state_dict` written by the
converter.

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
