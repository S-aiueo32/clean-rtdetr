# Penn-Fudan small detection benchmark

[Penn-Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) contains 170 real
images. Its archive is about 51 MB. Images and
annotations belong to the original dataset authors and are downloaded separately;
they are not included in the Apache-2.0 Python distribution. See the source page
for attribution and dataset information: Wang, Shi, Song, and Shen, *Object
Detection Combining Recognition and Segmentation*, ACCV 2007.

## Prepare

Run from the repository root (or worktree root):

```sh
uv sync --all-packages --extra benchmark
mkdir -p datasets
curl --fail --location https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip \
  --output datasets/PennFudanPed.zip
shasum -a 256 datasets/PennFudanPed.zip
unzip -q datasets/PennFudanPed.zip -d datasets
```

The archive verified for this integration has SHA-256
`9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`.
Check this value before extraction. `datasets/` is ignored by Git. The loader
expects `PNGImages/*.png` and matching `PedMasks/*_mask.png` beneath
`datasets/PennFudanPed`; it never downloads data automatically. Installed users
can install `clean-rtdetr-training[benchmark]` and pass their extracted directory.

The source page reports 345 pedestrians, but the archive with this checksum
contains 423 nonzero mask instances, matching the total object count in its
170 text annotation files. This loader uses those distributed masks.

## Dataset contract and comparison protocol

```python
from clean_rtdetr_training.data import PennFudanDetectionDataset

train = PennFudanDetectionDataset("datasets/PennFudanPed", split="train")
val = PennFudanDetectionDataset("datasets/PennFudanPed", split="val")
image, target = train[0]
```

- The default split sorts filenames, shuffles with a private PyTorch generator
  seeded with 41, and reserves the final 50 images for validation: 120 train,
  50 validation. This project split is not an official Penn-Fudan protocol.
  `image_paths` exposes the ordered members; save them with experiment results.
  Use the same `seed` and `val_size` for both splits. `split="all"` returns all
  images in filename order.
- Images use the core inference preprocessing: RGB, bilinear resize to 640×640,
  CPU float32 CHW values in [0, 1], without augmentation or mean/std normalization.
- Every nonzero mask ID becomes one box; zero is mask background. Person labels
  are **0**, with no separate background class. Boxes use normalized `cxcywh`
  relative to the original image, preserved by the direct square resize.
  Maxima are exclusive (`max_pixel + 1`), so single-pixel instances have nonzero
  area. This convention can differ from tutorials using inclusive maxima.
- Empty masks produce empty targets. Missing masks, invalid split parameters,
  and masks with inconsistent dimensions are rejected.

Record the source commit, archive checksum, split filenames, seed, complete
configuration, initialization weights, and metric definition when comparing
experiments. Keep validation images out of training.

## Training smoke run

```sh
uv run clean-rtdetr-train --config-dir configs --config-name penn_fudan_training \
  trainer.max_epochs=1 +trainer.limit_train_batches=1
```

The recipe sets one person class and uses the existing compact CPU model. It is
a wiring check, not a tuned accuracy baseline. Remove the batch limit to iterate
over the training split. Override `datamodule.dataset_factory.root` for another
extraction location. The Lightning CLI currently runs training only; the `val`
dataset is available to Python callers, but this recipe does not compute AP or
select checkpoints using validation. No convergence or AP result is claimed.
