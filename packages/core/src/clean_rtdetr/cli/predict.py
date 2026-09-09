# Copyright 2026 Clean RT-DETR contributors
"""Run image-file inference with explicitly verified official R50vd weights."""

import json
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images


@torch.inference_mode()
def run(
    image_paths: list[Path],
    checkpoint: Path,
    *,
    source: Literal["model", "ema"],
    expected_sha256: str,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Return all 300 predictions per image as JSON-compatible class-index records."""
    model = rtdetr_r50vd().eval()
    load_official_model_weights(
        checkpoint, model=model, source=source, expected_sha256=expected_sha256
    )
    model.to(device)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    batch = preprocess_images(images)
    output = model(batch.images.to(device), plan=DecoderPlan(collect_intermediate=False))
    detections = postprocess(output.final, batch.original_sizes_wh)
    return [
        {
            "image": str(path),
            "original_size_wh": size.tolist(),
            "labels": labels.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "boxes_xyxy": boxes.cpu().tolist(),
        }
        for path, size, labels, scores, boxes in zip(
            image_paths,
            batch.original_sizes_wh,
            detections.labels,
            detections.scores,
            detections.boxes_xyxy,
            strict=True,
        )
    ]


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(help="Input image files.")],
    checkpoint: Annotated[Path, typer.Option(help="Official checkpoint path.")],
    source: Annotated[Literal["model", "ema"], typer.Option(help="Checkpoint source.")],
    expected_sha256: Annotated[str, typer.Option(help="Expected checkpoint SHA-256.")],
    device: Annotated[str, typer.Option(help="Torch device.")] = "cpu",
) -> None:
    """Read image paths and print detections without downloading any artifacts."""
    results = run(
        images,
        checkpoint,
        source=source,
        expected_sha256=expected_sha256,
        device=device,
    )
    print(json.dumps(results, indent=2, allow_nan=False))


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
