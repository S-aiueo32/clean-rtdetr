# COCO bounding-box evaluation

Prediction generation and metric calculation are separate commands. Image
loading, preprocessing, model inference, category mapping, and result JSON
serialization do not import pycocotools. Only the private metric adapter
`evaluation/_pycocotools.py` loads it, when `evaluate_coco` is called. Its mutable
COCO objects and native-extension types never appear in the public API.

## Generate predictions

Install image support with `uv sync --locked --extra inference`, then use local
images, annotations, and the verified official checkpoint:

```sh
uv run python examples/coco_predictions.py \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --output outputs/coco-predictions.json
```

This writes predictions and `outputs/coco-predictions.report.json`. It reads
images in numeric image-ID order, one at a time, uses the existing RGB Pillow
640×640 preprocessing and all six decoder layers, and retains top-300 detections
without clipping, score thresholds, or NMS. CPU FP32 is the default; `--device`
selects another PyTorch device. No images or weights are downloaded.

`--limit N` selects the first N images for a smoke run. The report records the
actual selected IDs, image/checkpoint/annotation hashes, code and environment,
and whether the run covers all annotation images or a subset. Image dimensions
must match the annotations, and image paths must remain inside `--images`.
Existing output/report files are rejected.

`coco_predictions(detections, image_ids)` converts project-owned `Detections`
into plain records with `image_id`, `category_id`, `bbox` (absolute xywh), and
`score`. Label indices 0–79 map to the 80 non-contiguous COCO category IDs;
for example, label 11 maps to category 13. The order follows the
[pinned RT-DETR mapping](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/data/coco/coco_dataset.py).
Annotation category order is irrelevant, but its IDs must be exactly that set.

## Calculate AP from saved JSON

Only this step requires the evaluation extra. It can run in a separate
environment from inference:

```sh
uv sync --locked --extra evaluation
uv run python examples/coco_evaluation.py \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --predictions outputs/coco-predictions.json \
  --prediction-report outputs/coco-predictions.report.json \
  --report outputs/coco-metrics.json
```

The `evaluation` extra adds pycocotools and its transitive NumPy dependency.
The uv configuration disallows building pycocotools from source: an unsupported
environment fails to install this extra instead of invoking a C compiler.
Prediction generation remains available with the inference extra alone.

The optional prediction report is hash-checked against both input JSON files and
supplies the complete set of evaluated image IDs, including images with no
detections. Without it, **all annotated images** are evaluated; IDs are never
inferred from the predictions. Thus a partial prediction file cannot silently
exclude missed images from AP.

`evaluate_coco(annotations, predictions, image_ids=...)` accepts only a path,
plain prediction records, and optional image IDs. It returns a dictionary of
Python floats/`None`. The official bbox evaluator keeps its standard IoU and
recall thresholds, crowd/ignore handling, area ranges, and maxDets `[1,10,100]`.
This last setting is distinct from the model's top-300 postprocessing.
See the [official evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

The report expresses AP and AR in **0–100 units**. COCO's undefined values (for
example, a size range with no ground truth) become JSON `null`. Empty prediction
lists are supported, and the evaluator cannot mutate the caller's records.

## Validation scope

Offline tests use tiny synthetic annotations to check category conversion,
perfect/missing/empty predictions, image subsets, ignored crowd annotations,
and optional-dependency isolation. They establish the evaluation path, not the
model's COCO accuracy. Full COCO val2017 AP, comparison against upstream under
identical conditions, and GPU/latency measurements remain pending. A smoke
subset must not be reported as full COCO AP.
