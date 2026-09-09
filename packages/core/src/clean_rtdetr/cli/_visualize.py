# Copyright 2026 Clean RT-DETR contributors
"""Private Pillow rendering helpers for prediction CLI output."""

from pathlib import Path

from jaxtyping import Float, Int64
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

_COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _clamp(value: float, upper: int) -> float:
    return max(0.0, min(value, float(upper - 1)))


def _class_name(label: int) -> str:
    if 0 <= label < len(_COCO_CLASS_NAMES):
        return _COCO_CLASS_NAMES[label]
    return f"class {label}"


def _save_visualization(
    image: Image.Image,
    labels: Int64[Tensor, "detections"],  # noqa: F821 - jaxtyping shape name
    scores: Float[Tensor, "detections"],  # noqa: F821 - jaxtyping shape name
    boxes_xyxy: Float[Tensor, "detections 4"],
    path: Path,
) -> None:
    """Save an RGB copy of an image with its selected detections drawn."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    font = ImageFont.load_default(size=16)
    for label, score, box in zip(
        labels.detach().cpu().tolist(),
        scores.detach().cpu().tolist(),
        boxes_xyxy.detach().cpu().tolist(),
        strict=True,
    ):
        left, top, right, bottom = (
            _clamp(float(box[0]), width),
            _clamp(float(box[1]), height),
            _clamp(float(box[2]), width),
            _clamp(float(box[3]), height),
        )
        left, right = min(left, right), max(left, right)
        top, bottom = min(top, bottom), max(top, bottom)
        draw.rectangle((left, top, right, bottom), outline="red", width=2)
        text = f"{_class_name(int(label))} {float(score):.2f}"
        text_box = draw.textbbox((0, 0), text, font=font, anchor="lt")
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_top = top - text_height
        if text_top < 0:
            text_top = top
        text_left = max(0.0, min(left, width - text_width))
        text_top = max(0.0, min(text_top, height - text_height))
        draw.rectangle(
            (text_left, text_top, text_left + text_width, text_top + text_height),
            fill="red",
        )
        draw.text((text_left, text_top), text, font=font, fill="white", anchor="lt")
    canvas.save(path, format="PNG")
