# Copyright 2026 Clean RT-DETR contributors
"""Run image-file inference with native R50vd weights."""

import json
import math
from pathlib import Path
from typing import Annotated

import torch
import typer
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.cli._visualize import _save_visualization
from clean_rtdetr.inference import postprocess, preprocess_images


def _validate_conf_thresh(conf_thresh: float) -> None:
    if not isinstance(conf_thresh, (int, float)) or not math.isfinite(conf_thresh):
        msg = "conf_thresh must be a finite number in the range [0, 1]"
        raise ValueError(msg)
    if not 0 <= conf_thresh <= 1:
        msg = "conf_thresh must be in the range [0, 1]"
        raise ValueError(msg)


def _visualization_paths(image_paths: list[Path], visualize_dir: Path | None) -> tuple[Path, ...]:
    if visualize_dir is None:
        return ()
    destinations = tuple(
        (visualize_dir / f"{index:04d}-{path.stem}.png").resolve()
        for index, path in enumerate(image_paths, start=1)
    )
    source_set = {path.resolve() for path in image_paths}
    for destination in destinations:
        if destination in source_set:
            msg = f"visualization output would overwrite an input image: {destination}"
            raise ValueError(msg)
    visualize_dir.mkdir(parents=True, exist_ok=True)
    return destinations


@torch.inference_mode()
def run(
    image_paths: list[Path],
    checkpoint: Path,
    *,
    device: str = "cpu",
    conf_thresh: float = 0.0,
    visualize_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Run local native R50vd inference and return one JSON-compatible record per image.

    All six decoder layers run; top-300 query/class candidates are ranked before
    inclusive ``conf_thresh`` filtering. ``visualize_dir`` writes matching
    annotated PNGs and rejects an output that would overwrite an input.
    """
    _validate_conf_thresh(conf_thresh)
    if not image_paths:
        msg = "image_paths must contain at least one image"
        raise ValueError(msg)
    visualization_paths = _visualization_paths(image_paths, visualize_dir)

    model = rtdetr_r50vd().eval()
    load_model_weights(checkpoint, model=model)
    model.to(device)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    batch = preprocess_images(images)
    output = model(batch.images.to(device), plan=DecoderPlan(collect_intermediate=False))
    detections = postprocess(output.final, batch.original_sizes_wh)
    results: list[dict[str, object]] = []
    for index, (path, image, size, labels, scores, boxes) in enumerate(
        zip(
            image_paths,
            images,
            batch.original_sizes_wh,
            detections.labels,
            detections.scores,
            detections.boxes_xyxy,
            strict=True,
        ),
        start=1,
    ):
        selected = scores >= conf_thresh
        selected_labels = labels[selected]
        selected_scores = scores[selected]
        selected_boxes = boxes[selected]
        results.append(
            {
                "image": str(path),
                "original_size_wh": size.tolist(),
                "labels": selected_labels.cpu().tolist(),
                "scores": selected_scores.cpu().tolist(),
                "boxes_xyxy": selected_boxes.cpu().tolist(),
            }
        )
        if visualization_paths:
            _save_visualization(
                image,
                selected_labels,
                selected_scores,
                selected_boxes,
                visualization_paths[index - 1],
            )
    return results


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(help="Input image files.")],
    checkpoint: Annotated[Path, typer.Option(help="Native state_dict checkpoint path.")],
    device: Annotated[str, typer.Option(help="Torch device.")] = "cpu",
    conf_thresh: Annotated[
        float, typer.Option(help="Minimum detection confidence in the range [0, 1].")
    ] = 0.0,
    visualize_dir: Annotated[
        Path | None, typer.Option(help="Optional directory for annotated PNG images.")
    ] = None,
) -> None:
    """Print local image detections as JSON; no image or checkpoint is downloaded."""
    results = run(
        images,
        checkpoint,
        device=device,
        conf_thresh=conf_thresh,
        visualize_dir=visualize_dir,
    )
    print(json.dumps(results, indent=2, allow_nan=False))


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
