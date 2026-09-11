# Real-image R50vd training and resume

`tools/full_r50vd_training.py` trains the complete R50vd detector at 640×640
with 80 classes, 100 denoising queries, Hungarian matching, Varifocal/L1/GIoU
losses, AdamW, and EMA. Parameters are randomly initialized with seed 42;
training and shuffling use seed 41. This is an execution and resume recipe,
not the upstream augmentation schedule or a measured convergence result.
The named preset retains its frozen stem and frozen backbone normalization
(`freeze_at=0`, `freeze_norm=True`); the remaining backbone stages, encoder,
selector, decoder, and denoising embedding participate in training.

The dataset maps sorted COCO category IDs to contiguous labels, clips boxes
to image bounds, excludes crowd/invalid boxes and images without valid boxes,
and stretches RGB pixels to 640×640 with Pillow bilinear resampling. Pixels
are divided by 255. No random augmentation is applied. Checkpoints record the
annotation hash, selected image hashes, targets, and training settings; changing
these inputs on resume is rejected.

Install the checkout workspace with `uv sync --all-packages`. This acceptance
tool also reuses the checkout's exact-restoration test probe, so retain
`packages/training/tests` and install the development dependencies when moving
the job to Kaggle. Use a CUDA PyTorch wheel matching the remote driver. The
validated Kaggle environment installed torch 2.13.0+cu126 in an isolated Python
3.13 environment. When using the uv workspace, install the CUDA wheel after
syncing and keep `--no-sync` on subsequent runs so the chosen GPU environment
is preserved:

```sh
uv sync --all-packages
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

## Bounded free-tier validation

Run each precision in a new output directory:

```sh
uv run --no-sync python tools/full_r50vd_training.py validate \
  --image-root /path/to/val2017 \
  --annotation /path/to/annotations/instances_val2017.json \
  --output-dir /kaggle/working/r50vd-fp32 \
  --limit 4 --batch-size 2 --precision 32-true --tolerance 1e-4
uv run --no-sync python tools/full_r50vd_training.py validate \
  --image-root /path/to/val2017 \
  --annotation /path/to/annotations/instances_val2017.json \
  --output-dir /kaggle/working/r50vd-amp \
  --limit 4 --batch-size 2 --precision 16-mixed --tolerance 2e-3
```

Both precisions compare two uninterrupted epochs with one epoch plus a fresh
resume. AMP uses initial GradScaler scale 128. Only executed optimizer steps count as updates; at least three successful
updates are required. Each executed update must have finite, nonzero gradients
and finite parameters. The probe checks finite losses, changed parameters,
exact restoration of model/optimizer/EMA/CUDA RNG/scaler state before the next
batch, and continuation agreement for model, optimizer, EMA, scheduler, scaler,
and progress. CUDA continuation uses the stated tolerances; exact restoration
does not use a tolerance. This short probe uses validation images as training
inputs solely to exercise execution; it does not evaluate held-out quality.

## Resumable training session

Attach COCO train2017 and its annotations to the Kaggle session. Omit `--limit`
to include all images with valid non-crowd targets. The exact update count
depends on filtering and batch size. Start with a single epoch:

```sh
uv run --no-sync python tools/full_r50vd_training.py train \
  --image-root /path/to/train2017 \
  --annotation /path/to/annotations/instances_train2017.json \
  --output-dir /kaggle/working/r50vd-session-1 \
  --batch-size 2 --precision 16-mixed --epochs 1
```

Save the Kaggle output containing `checkpoints/last.ckpt`, then attach that
output as input to the next session. Keep the same checkout, dependencies,
data, batch size, and precision. Set `--epochs` to the cumulative target:

```sh
uv run --no-sync python tools/full_r50vd_training.py train \
  --image-root /path/to/train2017 \
  --annotation /path/to/annotations/instances_train2017.json \
  --output-dir /kaggle/working/r50vd-session-2 \
  --batch-size 2 --precision 16-mixed --epochs 2 \
  --resume /path/to/previous-output/checkpoints/last.ckpt
```

Checkpoints are written at completed epoch boundaries and include optimizer,
scheduler, EMA, AMP scaler, Lightning loop state, and Python/NumPy/PyTorch/CUDA/
loader RNG state. An interruption within an epoch resumes from the last saved
epoch. A session must finish at least one epoch to produce its first checkpoint;
the full-data epoch duration has not yet been measured. The submitted free-tier
job only runs the bounded validation above; full-data training is prepared for
separately scheduled sessions.

## Recorded T4 result

[Kaggle version 2](https://www.kaggle.com/code/souuchida/clean-rtdetr-full-training-20260911)
completed both precisions. The [report](../benchmarks/full-r50vd-training-t4.json)
records input/source hashes and the full run evidence. Both modes executed four
optimizer updates, including two after exact checkpoint restoration. AMP kept
scale 128 throughout; no attempted step was counted as a successful update.

| Precision | Model max absolute continuation difference | EMA max absolute difference | rtol / atol |
| --- | ---: | ---: | ---: |
| FP32 | 0.0000480078 | 0.0000479147 | 0.0001 |
| AMP FP16 | 0.00194734 | 0.00194454 | 0.002 |

A prior twelve-epoch AMP probe restored checkpoint state exactly but exceeded
0.002 tolerance in an optimizer moment after continuation (difference 0.00268616).
CUDA `grid_sample` backward is nondeterministic. The two-epoch result establishes
short execution and resume, not long-run numerical equivalence; tolerances were
not increased. Full-data convergence remains unmeasured.
