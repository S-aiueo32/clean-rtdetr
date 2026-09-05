# Copyright 2026 Clean RT-DETR contributors
"""Run image-file inference with explicitly verified official R50vd weights."""

import argparse
import json
from pathlib import Path
from typing import Literal, cast

import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, build_rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.data import preprocess_images
from clean_rtdetr.postprocessing import postprocess


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
    model = build_rtdetr_r50vd().eval()
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


def main() -> None:
    """Read image paths and print detections without downloading any artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", choices=("model", "ema"), required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = run(
        args.images,
        args.checkpoint,
        source=cast("Literal['model', 'ema']", args.source),
        expected_sha256=args.expected_sha256,
        device=args.device,
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
