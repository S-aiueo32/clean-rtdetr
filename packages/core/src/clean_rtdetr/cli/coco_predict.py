# Copyright 2026 Clean RT-DETR contributors
"""Create local COCO-format predictions with verified RT-DETR inference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
import typer
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.compatibility import load_official_model_weights
from clean_rtdetr.evaluation import CocoPrediction, coco_predictions, load_coco_images
from clean_rtdetr.inference import postprocess, preprocess_images


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
        raise typer.BadParameter(message)
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
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path): _hash(path) for path in sorted(source_root.rglob("*.py"))}
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


@dataclass(frozen=True)
class _PredictionCommandArgs:
    images: Path
    annotations: Path
    checkpoint: Path
    source: str
    expected_sha256: str
    output: Path
    report: Path | None
    limit: int | None
    device: str


def _resolve_paths(args: Any) -> tuple[Path, Path, Path, Path, Path]:  # noqa: ANN401
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
    return annotations, images_root, checkpoint, output, report


@torch.inference_mode()
def run(args: Any) -> dict[str, object]:  # noqa: ANN401
    """Generate predictions and a provenance report for selected COCO images."""
    annotations, images_root, checkpoint, output, report = _resolve_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    records = load_coco_images(annotations)
    selected = records if args.limit is None else records[: args.limit]
    root = images_root
    selected_ids: list[int] = []
    image_digest: list[dict[str, object]] = []
    predictions: list[CocoPrediction] = []
    torch.set_num_threads(4)
    model = rtdetr_r50vd().eval()
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


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def predict(  # noqa: PLR0913, PLR0917
    images: Annotated[Path, typer.Option(help="COCO image directory.")],
    annotations: Annotated[Path, typer.Option(help="COCO annotation JSON.")],
    checkpoint: Annotated[Path, typer.Option(help="Official checkpoint path.")],
    source: Annotated[Literal["model", "ema"], typer.Option(help="Checkpoint source.")],
    expected_sha256: Annotated[str, typer.Option(help="Expected checkpoint SHA-256.")],
    output: Annotated[Path, typer.Option(help="Prediction JSON output path.")],
    report: Annotated[Path | None, typer.Option(help="Optional provenance report path.")] = None,
    limit: Annotated[int | None, typer.Option(callback=_positive)] = None,
    device: Annotated[str, typer.Option(help="Torch device.")] = "cpu",
) -> None:
    """Parse arguments and write local COCO predictions."""
    args = _PredictionCommandArgs(
        images=images,
        annotations=annotations,
        checkpoint=checkpoint,
        source=source,
        expected_sha256=expected_sha256,
        output=output,
        report=report,
        limit=limit,
        device=device,
    )
    print(json.dumps(run(args)["predictions"], sort_keys=True))


def main() -> None:
    """Run the Typer command-line application."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
