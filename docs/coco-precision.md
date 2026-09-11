# COCO precision evaluation on T4

The corrected official R50vd EMA model was evaluated on all 5,000 COCO val2017
images in CUDA FP32 and FP16 autocast. Both runs used identical image bytes,
category mapping, 640×640 Pillow preprocessing, top-300 postprocessing without
NMS or clipping, and FP32 model weights/public predictions. TF32 was disabled.

| Precision | bbox AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| FP32 | 53.053617 | 71.218128 | 57.670752 |
| FP16 autocast | 53.073743 | 71.216628 | 57.682320 |

The measured AP difference is +0.020125 percentage points for autocast. This
single checkpoint/dataset comparison establishes neither a universal accuracy
improvement nor identical predictions. Independent upstream comparison, training
convergence, and other model presets are separate validations.

The [machine-readable record](../benchmarks/coco-precision-t4.json) contains the
source archive, official/native checkpoint and input hashes, full metrics,
prediction hashes, Python/PyTorch/CUDA versions, and the private Kaggle job URL.
Large prediction artifacts are retained in the job outputs, not in Git.

## Reproduce

From this checkout, install the workspace and optional core inference/evaluation
dependencies. On CUDA, using a converted native official EMA state dictionary:

```sh
uv sync --all-packages --all-extras
uv run python tools/coco_precision.py \
  --images /data/val2017 --annotations /data/annotations/instances_val2017.json \
  --checkpoint /data/official-ema.pt --expected-sha256 NATIVE_SHA256 \
  --output outputs/coco-precision
```

The tool requires a new output directory and exactly 5,000 annotation images by
default. It verifies matching image bytes/IDs between precision runs and writes
separate prediction/provenance/metric files plus a summary. `--expected-images`
can validate a different complete annotation set; that is not the recorded
COCO val2017 benchmark. The GPU run used Python 3.13 and torch 2.13.0+cu126,
separately from the macOS workspace lock; see the recorded dependency freeze hash.
