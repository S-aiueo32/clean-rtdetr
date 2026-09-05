# Inference I/O

Pillow is an optional dependency. Install image support with:

```sh
uv sync --locked --extra inference
```

## Python API

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan
from clean_rtdetr.data import preprocess_images
from clean_rtdetr.postprocessing import postprocess

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

`examples/image_inference.py` accepts one or more image paths, a local official
checkpoint, its selected source, and expected SHA-256. It prints all 300
detections per image as JSON and runs all six decoder layers by default. The
example does not download images or weights; `--device` defaults to `cpu`.
