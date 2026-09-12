# Copyright 2026 Clean RT-DETR contributors
"""Convert verified official R50vd inference weights to a native state_dict."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer

from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr.cli._official_checkpoint import load_official_model_weights

# Keep these defaults synchronized with benchmarks/baseline.json.
DEFAULT_SOURCE: Literal["model", "ema"] = "ema"
DEFAULT_SHA256: str = "dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45"

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def convert(
    checkpoint: Annotated[Path, typer.Option(help="Official checkpoint path.")],
    output: Annotated[Path, typer.Option(help="New native state_dict path.")],
    source: Annotated[
        Literal["model", "ema"], typer.Option(help="Official weight source.")
    ] = DEFAULT_SOURCE,
    expected_sha256: Annotated[
        str, typer.Option(help="Expected official SHA-256; defaults to the pinned R50vd release.")
    ] = DEFAULT_SHA256,
) -> None:
    """Convert a verified official R50vd envelope to a new native plain state dictionary.

    The default source is the pinned EMA release and its SHA-256. Existing output
    paths are rejected. Prints JSON with the output path and hash plus the load
    report.
    """
    if output.exists():
        message = f"Output already exists: {output}"
        raise FileExistsError(message)
    model = rtdetr_r50vd().eval().cpu()
    report = load_official_model_weights(
        checkpoint, model=model, source=source, expected_sha256=expected_sha256
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(model.state_dict(), stream)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "official_checkpoint_load": asdict(report),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    """Run the installed official-weight conversion command."""
    app(standalone_mode=False)


if __name__ == "__main__":
    main()
