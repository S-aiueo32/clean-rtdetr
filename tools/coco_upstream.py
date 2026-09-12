# Copyright 2026 Clean RT-DETR contributors
"""Run full COCO clean/upstream parity on identical CUDA inputs."""
# ruff: noqa: D103, E402, E501, EM102, FBT001, I001, PLR0915, PLR2004, PT018, S101, S310, TRY003, ANN401

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from typing import Any, cast
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(os.environ["GPU_CHECK_REPO"])
WORK = Path(os.environ.get("GPU_CHECK_OUTPUT", "/kaggle/working/clean-rtdetr-t4-coco-upstream"))
DATA = Path(os.environ.get("COCO_DATA", "/kaggle/temp/clean-rtdetr-coco-upstream/data/val2017"))
ANN = Path(
    os.environ.get(
        "COCO_ANNOTATIONS",
        "/kaggle/temp/clean-rtdetr-coco-upstream/data/annotations/instances_val2017.json",
    )
)
COCO_TEMP = Path(os.environ.get("COCO_TEMP", "/kaggle/temp/clean-rtdetr-coco-upstream"))
SHA = "dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45"
URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
ANNOTATION_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_SHA = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ANNOTATION_SHA = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages/core/tests"))
from official_oracle import backbone_oracle, hybrid_encoder_oracle
from oracle import full_transformer_oracle

from clean_rtdetr import DecoderPlan, DetectionOutput, rtdetr_r50vd
from clean_rtdetr.evaluation import coco_predictions, evaluate_coco, load_coco_images
from clean_rtdetr.inference import postprocess, preprocess_images
from tools.compatibility import load_official_model_weights


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def fetch(url: str, path: Path, expected: str) -> None:
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as source, part.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if hashlib.sha256(part.read_bytes()).hexdigest() != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: {url}")
    part.replace(path)


def ensure_coco() -> tuple[Path, Path]:
    image_zip = COCO_TEMP / "val2017.zip"
    annotation_zip = COCO_TEMP / "annotations_trainval2017.zip"
    fetch(IMAGE_URL, image_zip, IMAGE_SHA)
    fetch(ANNOTATION_URL, annotation_zip, ANNOTATION_SHA)
    data = COCO_TEMP / "data"
    images = data / "val2017"
    annotations = data / "annotations" / "instances_val2017.json"
    if not images.is_dir():
        with zipfile.ZipFile(image_zip) as archive:
            archive.extractall(data)
    if not annotations.is_file():
        with zipfile.ZipFile(annotation_zip) as archive:
            archive.extract("annotations/instances_val2017.json", data)
    return images, annotations


def oracle_model(checkpoint: Path) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    backbone = backbone_oracle(
        depth=50, variant="d", return_idx=[1, 2, 3], freeze_at=0, freeze_norm=True, pretrained=False
    ).eval()
    encoder = hybrid_encoder_oracle(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=[640, 640],
    ).eval()
    decoder = full_transformer_oracle(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        learnt_init_query=False,
        num_denoising=100,
        eval_spatial_size=[640, 640],
    ).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["ema"]["module"]
    for prefix, module in (("backbone", backbone), ("encoder", encoder), ("decoder", decoder)):
        module.load_state_dict(
            {
                k.removeprefix(prefix + "."): v
                for k, v in state.items()
                if k.startswith(prefix + ".")
            },
            strict=True,
        )
        module.cuda()
    return backbone, encoder, decoder


def clean_and_upstream(
    clean: Any,
    oracle: tuple[Any, ...],
    image: torch.Tensor,
    sizes: torch.Tensor,
    amp: bool,
) -> tuple:
    backbone, encoder, decoder = oracle
    ids_clean: list[torch.Tensor] = []
    hook_clean = clean.selector.register_forward_hook(
        lambda _m, _a, output: ids_clean.append(output.queries.source_indices.detach().cpu())
    )
    ids_upstream: list[torch.Tensor] = []
    hook_upstream = decoder.enc_score_head.register_forward_hook(
        lambda _m, _a, output: ids_upstream.append(
            output.max(-1).values.topk(300, dim=1).indices.cpu()
        )
    )
    context = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    try:
        with torch.inference_mode(), context:
            clean_output = clean(image, plan=DecoderPlan(collect_intermediate=False)).final
            features = encoder(backbone(image))
            upstream_output = decoder(features)
        clean_output = DetectionOutput(clean_output.logits.float(), clean_output.boxes.float())
        upstream_output = DetectionOutput(
            upstream_output["pred_logits"].float(), upstream_output["pred_boxes"].float()
        )
        clean_det = coco_predictions(
            postprocess(clean_output, sizes.to("cuda"), top_k=300),
            torch.tensor([0], dtype=torch.int64),
        )
        upstream_det = coco_predictions(
            postprocess(upstream_output, sizes.to("cuda"), top_k=300),
            torch.tensor([0], dtype=torch.int64),
        )
        return (
            clean_det,
            upstream_det,
            ids_clean[0][0].tolist(),
            ids_upstream[0][0].tolist(),
            clean_output,
            upstream_output,
        )
    finally:
        hook_clean.remove()
        hook_upstream.remove()


def main() -> None:
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    global DATA, ANN
    DATA, ANN = ensure_coco()
    checkpoint = ROOT.parent / "official.pth"
    urllib.request.urlretrieve(URL, checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == SHA
    records = load_coco_images(ANN)
    assert len(records) == 5000
    WORK.mkdir(parents=True, exist_ok=True)
    clean = rtdetr_r50vd().eval()
    load_official_model_weights(checkpoint, model=clean, source="ema", expected_sha256=SHA)
    clean.cuda()
    oracle = oracle_model(checkpoint)
    all_results: dict[str, object] = {}
    for precision, amp in (("fp32", False), ("amp_fp16", True)):
        clean_predictions = []
        upstream_predictions = []
        image_digests = []
        clean_ids = []
        upstream_ids = []
        raw_diffs = []
        for index, record in enumerate(records):
            data = (DATA / record.file_name).read_bytes()
            with Image.open(BytesIO(data)) as source:
                assert source.size == (record.width, record.height)
                batch = preprocess_images([source.convert("RGB")])
            result = clean_and_upstream(
                clean, oracle, batch.images.cuda(), batch.original_sizes_wh, amp
            )
            cp, up, ci, ui, co, uo = result
            for prediction in cp + up:
                prediction["image_id"] = record.image_id
            clean_predictions.extend(cp)
            upstream_predictions.extend(up)
            image_digests.append(
                {"image_id": record.image_id, "sha256": hashlib.sha256(data).hexdigest()}
            )
            clean_ids.append(ci)
            upstream_ids.append(ui)
            clean_pos = {query_id: position for position, query_id in enumerate(ci)}
            upstream_pos = {query_id: position for position, query_id in enumerate(ui)}
            common = sorted(clean_pos.keys() & upstream_pos.keys())
            raw_diffs.append(
                {
                    "max_abs_logits_ordered": (co.logits - uo.logits).abs().max().item(),
                    "max_abs_boxes_ordered": (co.boxes - uo.boxes).abs().max().item(),
                    "common_query_count": len(common),
                    "max_abs_logits_query_aligned": (
                        co.logits[:, [clean_pos[x] for x in common]]
                        - uo.logits[:, [upstream_pos[x] for x in common]]
                    )
                    .abs()
                    .max()
                    .item()
                    if common
                    else None,
                    "max_abs_boxes_query_aligned": (
                        co.boxes[:, [clean_pos[x] for x in common]]
                        - uo.boxes[:, [upstream_pos[x] for x in common]]
                    )
                    .abs()
                    .max()
                    .item()
                    if common
                    else None,
                }
            )
            if (index + 1) % 250 == 0:
                print(precision, index + 1, flush=True)
        clean_metrics = evaluate_coco(
            ANN, clean_predictions, image_ids=[r.image_id for r in records]
        )
        upstream_metrics = evaluate_coco(
            ANN, upstream_predictions, image_ids=[r.image_id for r in records]
        )
        prediction_artifacts = {}
        for name, predictions in (("clean", clean_predictions), ("upstream", upstream_predictions)):
            artifact = WORK / f"{name}-{precision}-predictions.json"
            artifact.write_text(json.dumps(predictions, separators=(",", ":")) + "\n")
            prediction_artifacts[name] = {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "count": len(predictions),
            }
        exact = [a == b for a, b in zip(clean_ids, upstream_ids, strict=True)]
        same_sets = [set(a) == set(b) for a, b in zip(clean_ids, upstream_ids, strict=True)]
        common_counts = [len(set(a) & set(b)) for a, b in zip(clean_ids, upstream_ids, strict=True)]
        metric_deltas = {
            key: float(cast("float", clean_metrics[key]))
            - float(cast("float", upstream_metrics[key]))
            for key in clean_metrics
            if isinstance(clean_metrics[key], (int, float))
            and isinstance(upstream_metrics.get(key), (int, float))
        }
        all_results[precision] = {
            "scope": "all_annotation_images",
            "image_count": len(records),
            "input_order_digest": digest(image_digests),
            "image_sha256_digest": digest(image_digests),
            "clean_metrics": clean_metrics,
            "upstream_metrics": upstream_metrics,
            "clean_minus_upstream_metric_deltas": metric_deltas,
            "subset_metrics": None,
            "prediction_artifacts": prediction_artifacts,
            "query_alignment": {
                "image_count": len(exact),
                "ordered_ids_equal_images": sum(exact),
                "same_query_set_images": sum(same_sets),
                "common_query_count_min": min(common_counts),
                "common_query_count_max": max(common_counts),
                "query_count": 300,
                "clean_ordered_ids_digest": digest(clean_ids),
                "upstream_ordered_ids_digest": digest(upstream_ids),
                "all_ordered_ids_equal": all(exact),
                "subset_diagnostic": "No subset inference or subset AP was run; this report contains full-scope query alignment only.",
            },
            "raw_output_diff_summary": {
                "max_abs_logits_ordered": max(
                    float(cast("float", x["max_abs_logits_ordered"])) for x in raw_diffs
                ),
                "max_abs_boxes_ordered": max(
                    float(cast("float", x["max_abs_boxes_ordered"])) for x in raw_diffs
                ),
                "max_abs_logits_query_aligned": max(
                    x["max_abs_logits_query_aligned"]
                    for x in raw_diffs
                    if x["max_abs_logits_query_aligned"] is not None
                ),
                "max_abs_boxes_query_aligned": max(
                    x["max_abs_boxes_query_aligned"]
                    for x in raw_diffs
                    if x["max_abs_boxes_query_aligned"] is not None
                ),
            },
        }
    report = {
        "schema_version": 1,
        "status": "completed",
        "scope": "all_annotation_images",
        "image_count": 5000,
        "annotations_sha256": hashlib.sha256(ANN.read_bytes()).hexdigest(),
        "preprocessing": {
            "implementation": "clean_rtdetr.inference.preprocess_images",
            "resize": "640x640",
            "uint8_rgb": True,
        },
        "postprocessing": {
            "top_k": 300,
            "category_map": "clean_rtdetr.evaluation.COCO_CATEGORY_IDS",
            "nms": False,
            "clipping": False,
        },
        "checkpoint_sha256": SHA,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": all_results,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "upstream-comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
