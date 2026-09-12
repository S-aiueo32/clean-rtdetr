# COCO bounding-box evaluation

English | [日本語](ja/evaluation.md)

Use these commands to create COCO prediction JSON from local images, then
calculate bbox metrics. Prediction generation and metric calculation are
separate so image inference does not require pycocotools.

## Prerequisites

Use Python 3.13 or newer, a converted native R50vd checkpoint, a COCO image
directory, and its annotation JSON. Convert official weights as described in
the [inference guide](inference.md). Neither command downloads images or
weights.

Prediction generation needs Pillow:

```sh
python -m pip install "clean-rtdetr[inference]"
```

Metric calculation needs the evaluation extra and can run in another
environment:

```sh
python -m pip install "clean-rtdetr[evaluation]"
```

The extra provides pycocotools and NumPy. In a checkout, the uv configuration
does not build pycocotools from source, so `uv sync --package clean-rtdetr
--extra evaluation` fails on an unsupported environment instead of invoking a C
compiler. This restriction does not apply to the `pip` command above.

<a id="generate-predictions"></a>

## Generate prediction JSON

Run the command with the directory containing COCO image files and the matching
annotation JSON:

```sh
clean-rtdetr-coco-predict \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --output outputs/coco-predictions.json
```

It writes the predictions and `outputs/coco-predictions.report.json`. Images are
processed one at a time in numeric image-ID order using RGB Pillow 640×640
preprocessing, all six decoder layers, and top-300 detection postprocessing.
There is no clipping, NMS, or score threshold. CPU FP32 is the default;
`--device` selects a PyTorch device.

Use `--limit N` only for a smoke run. The report records the selected image IDs,
input hashes, environment, and whether the selection covers all annotation
images. Image dimensions must agree with annotations, and filenames must remain
under `--images`. Existing output and report paths are rejected.

`coco_predictions(detections, image_ids)` converts project `Detections` to
records containing `image_id`, `category_id`, absolute `xywh` `bbox`, and
`score`. Labels 0–79 map to the 80 non-contiguous COCO IDs; label 11, for
example, maps to category 13. Annotation category order does not matter, but
its IDs must be exactly that set. The mapping follows the [pinned RT-DETR
source](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/data/coco/coco_dataset.py).

<a id="calculate-ap-from-saved-json"></a>

## Calculate metrics from saved predictions

Pass the prediction report whenever the prediction run was limited or otherwise
selected a subset:

```sh
clean-rtdetr-coco-evaluate \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --predictions outputs/coco-predictions.json \
  --prediction-report outputs/coco-predictions.report.json \
  --report outputs/coco-metrics.json
```

The report hashes are checked against both input JSON files and provide the full
evaluated image-ID set, including images with no detections. Without
`--prediction-report`, every annotated image is evaluated; the evaluator never
infers IDs from prediction rows. That prevents a partial prediction file from
silently excluding missed images.

`evaluate_coco(annotations, predictions, image_ids=...)` accepts an annotation
path, plain prediction records, and optional IDs. It returns Python floats or
`None` and does not mutate caller records. The standard COCO bbox evaluator
keeps its IoU/recall thresholds, crowd and ignore handling, area ranges, and
`maxDets=[1,10,100]`; this is separate from the model's top-300 candidates.
Metrics in the JSON report use a 0–100 scale, and undefined COCO values become
`null`. Empty prediction lists are valid.

<a id="validation-scope"></a>

## What has been measured

Tests use small synthetic annotations to exercise conversion, missing and empty
predictions, subsets, crowd annotations, and optional-dependency isolation.
They do not establish model accuracy. A separate full CPU FP32 run with official
pretrained R50vd EMA weights measured 53.05 AP, 71.22 AP50, and 57.67 AP75 on
all 5,000 COCO val2017 images; see the [result](../benchmarks/coco-val2017.md)
and [provenance report](../benchmarks/coco-val2017.json). A smoke subset is not
full COCO AP. Upstream comparison under identical conditions and GPU/latency
measurements remain pending.
