# Sample image provenance

`cat.png` is the unedited `chelsea.png` sample image from scikit-image.

- Source: [scikit-image v0.24.0, `chelsea.png`](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/chelsea.png)
- Photographer: Stéfan van der Walt
- Image license: [Creative Commons CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), as documented by [scikit-image](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.chelsea)
- SHA-256: `596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb`

The image license is separate from this repository's Apache license.

## Detection preview

`cat-prediction.png` is the same photograph with bounding boxes and labels drawn
by `clean-rtdetr-predict`. It was generated with the pinned official R50vd EMA
weights in CPU FP32, `--conf-thresh 0.5`, and `--visualize-dir outputs/quickstart`.
The source photograph remains CC0. The README quickstart reproduces the
prediction and writes the annotated image to `outputs/0001-cat.png`.
