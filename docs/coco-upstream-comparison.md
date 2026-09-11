# Pinned upstream COCO comparison

`tools/coco_upstream.py` runs the independent, hash-pinned RT-DETR oracle and
the clean implementation over all 5,000 COCO `val2017` images. It downloads and
verifies the official image and annotation archives, uses the clean
Pillow/uint8 640x640 preprocessing and COCO category mapping, and records the
official checkpoint SHA-256. FP32 and CUDA AMP FP16 are run at the same scope.

FP32 AP is 53.053604 for both implementations. AMP AP is 53.073743 for clean
and 53.073641 upstream. FP32 raw maxima are zero; AMP raw maxima are 0.0546875
for logits and 0.004773974 for boxes. All 5,000 images have identical ordered
300-query IDs. No subset AP is included.

Full prediction hashes, image digest, raw diffs, and source/job provenance are
in [`coco-upstream-comparison.json`](../benchmarks/coco-upstream-comparison.json).
The completed Kaggle job is
[`souuchida/clean-rtdetr-t4-coco-upstream-comparison`](https://www.kaggle.com/code/souuchida/clean-rtdetr-t4-coco-upstream-comparison).

Reproduce in a prepared GPU checkout with:

```sh
GPU_CHECK_REPO=/path/to/clean-rtdetr python tools/coco_upstream.py
```

The Kaggle bootstrap and generated source archive are kept as local submission
inputs and excluded from version control.
