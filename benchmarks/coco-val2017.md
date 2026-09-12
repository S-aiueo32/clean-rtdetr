# COCO val2017 CPU FP32 benchmark

On 2026-09-10, the official pretrained R50vd EMA weights achieved
**53.05 bbox AP** on all 5,000 COCO `val2017` images using native PyTorch
CPU FP32 inference. The [machine-readable result](coco-val2017.json) records
the inputs, source hashes, environment, and all twelve COCO metrics.

## Scope

- Model: official pretrained RT-DETR R50vd, EMA source, 80 COCO classes.
- Device and dtype: CPU, FP32, four Torch threads.
- Dataset: all 5,000 COCO `val2017` images and the matching 80-class
  `instances_val2017.json` annotations.
- Prediction path: native converted checkpoint, fixed 640x640 RGB Pillow bilinear preprocessing
  and division by 255, all six decoder layers, top-300 sigmoid detections with
  no score threshold, clipping, or NMS, and standard COCO bbox maxDets `[1, 10, 100]`.
- This benchmark does not claim an upstream same-run AP comparison, training
  reproduction, latency measurement, GPU validation, or exported-model accuracy.

## Pinned inputs

The source commit is `556de56fd7eb27b7f56b03515baaef36afb63e05`.

The official dataset archives were obtained from:

- `https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip`
- `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`

The official checkpoint is the pinned EMA artifact from `benchmarks/baseline.json`:

- URL: `https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth`
- SHA256: `dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45`
- Converted native checkpoint SHA256:
  `2663b90b77400da1f89ba4f5c1024261ceba1567a105d9aa161164e44eec3ae3`

## Environment and commands

The measured environment used macOS 26.6.1 arm64, Python 3.14.7,
PyTorch 2.14.0, Pillow 12.3.0, NumPy 2.5.2, and pycocotools 2.0.11.
Inference loads all weights from the official EMA checkpoint and uses no
stochastic augmentation. The source worktree was clean during prediction.

Run from the repository root. Install the workspace and COCO evaluation dependency:

```sh
uv sync --all-packages --extra evaluation
```

Download and extract the fixed input artifacts (about 1.2 GB of downloads):

```sh
curl --fail --location --create-dirs \
  --output datasets/coco/val2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
curl --fail --location --create-dirs \
  --output datasets/coco/annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q datasets/coco/val2017.zip -d datasets/coco
unzip -q datasets/coco/annotations_trainval2017.zip \
  annotations/instances_val2017.json -d datasets/coco
curl --fail --location --create-dirs \
  --output checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

The archive SHA-256 values and the extracted annotation SHA-256 are recorded in
[coco-val2017.json](coco-val2017.json). Compare downloaded artifacts with those
values before a reproduction run. The converter verifies the official checkpoint
against its pinned SHA-256.

Convert the official envelope to the native state dictionary:

```sh
.venv/bin/clean-rtdetr-convert \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --output checkpoints/rtdetr_r50vd.pth
```

Generate predictions for every annotated image:

```sh
.venv/bin/clean-rtdetr-coco-predict \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --expected-sha256 2663b90b77400da1f89ba4f5c1024261ceba1567a105d9aa161164e44eec3ae3 \
  --output outputs/coco-val2017-predictions.json \
  --device cpu
```

Evaluate using the matching prediction provenance report:

```sh
.venv/bin/clean-rtdetr-coco-evaluate \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --predictions outputs/coco-val2017-predictions.json \
  --prediction-report outputs/coco-val2017-predictions.report.json \
  --report outputs/coco-val2017-metrics.json
```

## Results

AP and AR use the 0–100 scale.

| Metric | Value |
| --- | ---: |
| Images evaluated | 5,000 |
| Predictions | 1,500,000 |
| AP@[.50:.95] | 53.05 |
| AP50 | 71.22 |
| AP75 | 57.67 |
| AP small | 34.73 |
| AP medium | 57.73 |
| AP large | 70.10 |
| AR100 | 72.14 |

The compact evidence file is [coco-val2017.json](coco-val2017.json). It records the
source revision, environment, dataset and checkpoint hashes, prediction and
metric report hashes, and the evaluation scope. Full prediction JSON and raw
manifests remain under the ignored `outputs/` directory and are not committed.
The prediction report contains all image IDs and per-image hashes. The recorded
result verifies that these IDs exactly match all 5,000 annotation images and
that the prediction, annotation, and metric-report hashes agree. Use new
checkpoint/output paths when repeating commands; existing outputs are rejected.

## Interpretation

These metrics establish the clean
R50vd native PyTorch CPU FP32 evaluation result for the pinned official EMA
checkpoint and the complete COCO `val2017` image set. They do not establish
upstream same-run equivalence, training convergence, GPU or latency behavior,
or ONNX/export accuracy.
