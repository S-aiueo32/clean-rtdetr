# Copyright 2026 Clean RT-DETR contributors
"""Create local COCO-format predictions with verified RT-DETR inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import torch
from hydra.utils import instantiate
from PIL import Image

from clean_rtdetr import DecoderPlan
from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.configuration import compose_model_config
from clean_rtdetr.data import preprocess_images
from clean_rtdetr.evaluation import CocoPrediction, coco_predictions, load_coco_images
from clean_rtdetr.postprocessing import postprocess


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _git(*args: str, fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(("git", *args), text=True).strip()  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        message = "must be positive"
        raise argparse.ArgumentTypeError(message)
    return parsed


def _provenance(  # noqa: PLR0913, PLR0917
    annotations: Path,
    checkpoint: Path,
    expected_sha256: str,
    source: str,
    image_digest: list[dict[str, object]],
    selected_ids: list[int],
    total_count: int,
    device: str,
) -> dict[str, object]:
    source_hashes = {
        str(path): _hash(path) for path in sorted(Path("src/clean_rtdetr").rglob("*.py"))
    }
    lockfile = Path("uv.lock")
    ordered = _hash_bytes(json.dumps(image_digest, separators=(",", ":"), sort_keys=True).encode())
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _hash(checkpoint),
            "expected_sha256": expected_sha256,
            "source": source,
        },
        "annotations": {"path": str(annotations), "sha256": _hash(annotations)},
        "input_images": {
            "ordered_digest": ordered,
            "selected_ids": selected_ids,
            "total_count": total_count,
            "selected_count": len(selected_ids),
            "images": image_digest,
        },
        "scope": "all_annotation_images" if len(selected_ids) == total_count else "subset",
        "preprocessing": {"resize": "640x640", "uint8_rgb": True},
        "postprocessing": {"top_k": 300, "nms": False, "clipping": False},
        "git_revision": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain", fallback="")),
        "git_diff_sha256": _hash_bytes(_git("diff", "HEAD", fallback="").encode()),
        "implementation_sha256": source_hashes,
        "cli_sha256": _hash(Path(__file__).resolve()),
        "uv_lock_sha256": _hash(lockfile) if lockfile.is_file() else "unavailable",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {"torch": torch.__version__, "pillow": _version("pillow")},
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    """Generate predictions and a provenance report for selected COCO images."""
    annotations = args.annotations.resolve()
    images_root = args.images.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    report = (args.report or args.output.with_suffix(".report.json")).resolve()
    for path, name in ((output, "output"), (report, "report")):
        if path.exists():
            message = f"{name} already exists: {path}"
            raise FileExistsError(message)
    if (
        output == report
        or output in {annotations, checkpoint}
        or report in {annotations, checkpoint}
    ):
        message = "output and report paths must not collide with inputs"
        raise ValueError(message)
    if not annotations.is_file() or not images_root.is_dir() or not checkpoint.is_file():
        message = "annotations, images, and checkpoint paths must exist"
        raise FileNotFoundError(message)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    records = load_coco_images(annotations)
    selected = records if args.limit is None else records[: args.limit]
    root = images_root
    selected_ids: list[int] = []
    image_digest: list[dict[str, object]] = []
    predictions: list[CocoPrediction] = []
    torch.set_num_threads(4)
    model = instantiate(compose_model_config("r50vd"), _convert_="all").eval()
    load_official_model_weights(
        checkpoint,
        model=model,
        source=cast("Literal['model', 'ema']", args.source),
        expected_sha256=args.expected_sha256,
    )
    model.to(args.device)
    for record in selected:
        image_path = (root / record.file_name).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            message = f"Image filename escapes images root: {record.file_name}"
            raise ValueError(message) from error
        data = image_path.read_bytes()
        with Image.open(BytesIO(data)) as image:
            if image.size != (record.width, record.height):
                message = f"Image dimensions do not match annotation: {record.file_name}"
                raise ValueError(message)
            image_rgb = image.convert("RGB")
        batch = preprocess_images([image_rgb])
        output_model = model(
            batch.images.to(args.device), plan=DecoderPlan(collect_intermediate=False)
        )
        detections = postprocess(output_model.final, batch.original_sizes_wh, top_k=300)
        predictions.extend(coco_predictions(detections, torch.tensor([record.image_id])))
        selected_ids.append(record.image_id)
        image_digest.append({"image_id": record.image_id, "sha256": _hash_bytes(data)})
    _write(output, predictions)
    report_data = {
        "schema_version": 1,
        "dtype": "float32",
        "decoder_layers": 6,
        "predictions": {"path": str(output), "sha256": _hash(output), "count": len(predictions)},
        **_provenance(
            annotations,
            checkpoint,
            args.expected_sha256,
            args.source,
            image_digest,
            selected_ids,
            len(records),
            str(args.device),
        ),
    }
    _write(report, report_data)
    return report_data


def main() -> None:
    """Parse arguments and write local COCO predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", choices=("model", "ema"), required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args)["predictions"], sort_keys=True))


if __name__ == "__main__":
    main()
