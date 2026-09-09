# Copyright 2026 Clean RT-DETR contributors
"""Construct contrastive denoising queries for RT-DETR training.

Adapted from lyuwenyu/RT-DETR (Apache-2.0); the pinned source and license are
recorded under ``third_party/RT-DETR``. This version adds typed contracts,
validation, and float64 support while retaining upstream FP32 RNG ordering.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from jaxtyping import (  # noqa: TC002 - runtime jaxtyping instrumentation resolves annotations
    Bool,
    Float,
    Int32,
    Int64,
)
from torch import Tensor, nn

from clean_rtdetr.contracts import DenoisingQueries
from clean_rtdetr.geometry import inverse_sigmoid
from clean_rtdetr_training.contracts import (
    DenoisingBatch,
    DenoisingMetadata,
    DetectionTarget,
    _validate_target,
)
from clean_rtdetr_training.interfaces import DenoisingGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DenoisingBatch", "DenoisingConfig", "DenoisingQueryGenerator"]


@dataclass(frozen=True)
class DenoisingConfig:
    """Noise levels and the target count used to choose the number of DN groups.

    ``num_denoising`` chooses how many repetitions of the largest target set
    to make; it is neither the number of output queries nor a hard upper bound
    on them. Zero disables DN. For enabled DN and a nonempty target batch,
    with ``M`` equal to the largest target count, the generator uses
    ``G = max(1, num_denoising // M)`` groups, each containing ``2 * M``
    slots (M positive and M negative), for ``D = 2 * M * G`` DN queries. Thus
    ``num_denoising=100`` and ``M=20`` produce ``G=5`` and ``D=200``.
    """

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject settings outside the upstream operation's useful domain."""
        if (
            isinstance(self.num_denoising, bool)
            or not isinstance(self.num_denoising, int)
            or self.num_denoising < 0
        ):
            msg = "num_denoising must be a nonnegative integer"
            raise ValueError(msg)
        if not math.isfinite(self.label_noise_ratio) or not 0 <= self.label_noise_ratio <= 1:
            msg = "label_noise_ratio must be between 0 and 1"
            raise ValueError(msg)
        if not math.isfinite(self.box_noise_scale) or self.box_noise_scale < 0:
            msg = "box_noise_scale must be nonnegative"
            raise ValueError(msg)


@dataclass(frozen=True)
class _PreparedQueries:
    """Named tensors produced while assembling the padded DN query groups.

    ``negative_indicator`` is a floating ``0``/``1`` tensor because it is
    multiplied into the sampled perturbation magnitudes.  It is distinct from
    the boolean attention mask, where ``True`` means that attention is blocked.
    """

    labels: Int32[Tensor, "batch queries"]
    boxes_cxcywh: Float[Tensor, "batch queries 4"]
    valid_target_mask: Bool[Tensor, "batch queries"]
    negative_indicator: Float[Tensor, "batch queries 1"]
    # The underscore permits different match counts for each image.
    positive_indices: tuple[Int64[Tensor, " _matches"], ...]


def _box_cxcywh_to_xyxy(
    boxes: Float[Tensor, "*batch 4"],
) -> Float[Tensor, "*batch 4"]:
    """Convert center-size boxes to corner coordinates."""
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _box_xyxy_to_cxcywh(
    boxes: Float[Tensor, "*batch 4"],
) -> Float[Tensor, "*batch 4"]:
    """Convert corner boxes to center-size coordinates."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


class DenoisingQueryGenerator(DenoisingGenerator):
    """Build padded positive/negative target groups for contrastive DN training.

    For a batch padded to ``M = max(target counts)``, each of ``G`` groups
    contains M positive slots followed by M negative slots. An image with m
    targets uses the first m slots of each half; the rest are padding. The
    decoder receives ``2 * M * G`` DN slots. Positive slots retain the original
    target assignment for supervision even when their input labels are corrupted;
    negative slots receive stronger box perturbations when noise is enabled. The
    returned ``positive_indices`` are the flattened positive slots for each
    image and are used by the criterion instead of running Hungarian matching.

    Decoder self-attention uses rows for query positions and columns for key
    positions (``X = True`` blocks attention; ``. = False`` allows it)::

        query / key     DN group 0   DN group 1   matching
        DN group 0          .            X           .
        DN group 1          X            .           .
        matching            X            X           .

    A DN group can attend its own group and matching queries, while groups are
    isolated from one another.  Matching queries cannot read DN queries.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        config: DenoisingConfig | None = None,
    ) -> None:
        """Create the padding-aware class embedding and store validated dimensions."""
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            msg = "num_classes must be a positive integer"
            raise ValueError(msg)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            msg = "hidden_dim must be a positive integer"
            raise ValueError(msg)
        if config is not None and not isinstance(config, DenoisingConfig):
            msg = "config must be a DenoisingConfig"
            raise TypeError(msg)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.config = config if config is not None else DenoisingConfig()
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the learned query content."""
        return self.class_embed.weight.dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the learned query content."""
        return self.class_embed.weight.device

    def checkpoint_config(self) -> dict[str, object]:
        """Describe query dimensions and noise settings for checkpoint resumes."""
        return {
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "config": asdict(self.config),
        }

    def forward(
        self,
        targets: Sequence[DetectionTarget],
        *,
        num_queries: int,
    ) -> DenoisingBatch | None:
        """Return padded DN queries, or ``None`` when there is no DN work.

        ``num_queries`` is the number ``Q`` of ordinary matching queries
        that will follow the DN prefix. It is separate from the generated DN count
        ``D = 2 * M * G``: the returned attention mask has shape
        ``(D + Q, D + Q)``, and ``metadata.num_queries`` is ``D``. If DN is
        disabled or every image has zero targets, this method returns ``None``.

        The construction order is part of the pinned FP32 behavior: validate
        and pad targets, sample label noise, sample box noise, embed labels,
        then construct the attention mask.  In particular, label-noise draws
        happen before box-noise draws, so changing this order changes the
        seeded random stream even when the resulting shapes are unchanged.
        """
        weight = self.class_embed.weight
        self._validate_inputs(targets, num_queries, weight)
        if self.config.num_denoising == 0:
            return None
        target_counts = [int(target.labels.shape[0]) for target in targets]
        max_targets = max(target_counts)
        if max_targets == 0:
            return None

        num_groups = max(1, self.config.num_denoising // max_targets)
        prepared = self._prepare_queries(targets, target_counts, max_targets, num_groups, weight)
        labels = self._apply_label_noise(prepared.labels, prepared.valid_target_mask)
        boxes_cxcywh = self._apply_box_noise(prepared.boxes_cxcywh, prepared.negative_indicator)
        if self.config.box_noise_scale > 0:
            reference_logits = inverse_sigmoid(boxes_cxcywh)
        else:
            # Compatibility exception: the zero-noise path historically passes
            # normalized coordinates to the decoder instead of box logits.
            reference_logits = boxes_cxcywh
        content = self.class_embed(labels)
        actual_queries = max_targets * 2 * num_groups
        attention_mask = self._attention_mask(
            actual_queries, num_queries, max_targets, num_groups, weight.device
        )

        return DenoisingBatch(
            queries=DenoisingQueries(content, reference_logits, attention_mask),
            metadata=DenoisingMetadata(
                positive_indices=prepared.positive_indices,
                num_groups=num_groups,
                num_queries=actual_queries,
            ),
        )

    def _validate_inputs(
        self,
        targets: Sequence[DetectionTarget],
        num_queries: int,
        weight: Float[Tensor, "classes hidden"],
    ) -> None:
        """Validate the batch boundary before inspecting or sampling targets."""
        if isinstance(num_queries, bool) or not isinstance(num_queries, int) or num_queries <= 0:
            msg = "num_queries must be a positive integer"
            raise ValueError(msg)
        if not targets:
            msg = "Denoising targets must contain at least one batch image"
            raise ValueError(msg)
        self._validate_targets(targets, weight)

    def _validate_targets(
        self,
        targets: Sequence[DetectionTarget],
        weight: Float[Tensor, "classes hidden"],
    ) -> None:
        """Check target types, labels, and boxes against the query embedding's dtype/device."""
        for target in targets:
            if not isinstance(target, DetectionTarget):
                msg = "targets must contain DetectionTarget values"
                raise TypeError(msg)
            _validate_target(
                target,
                classes=self.num_classes,
                dtype=weight.dtype,
                device=weight.device,
            )

    def _prepare_queries(
        self,
        targets: Sequence[DetectionTarget],
        target_counts: list[int],
        max_targets: int,
        num_groups: int,
        weight: Float[Tensor, "classes hidden"],
    ) -> _PreparedQueries:
        """Pad targets, duplicate groups, and record positive slot indices.

        Each image contributes ``m`` real targets and ``M-m`` padding entries;
        padding uses the extra embedding index ``num_classes`` and is excluded
        from label replacement. Random draws still cover the padded tensor;
        zero-size padded boxes have zero perturbation magnitude. After tiling,
        the slot order is
        ``[positive_0:M, negative_0:M]`` for each group.  Thus image ``b``'s
        positive slots are exactly
        ``g * (2*M) + target_index`` for ``g = 0 .. G-1``; those flattened
        indices are returned for DN loss supervision.
        """
        device, dtype = weight.device, weight.dtype
        batch_size = len(targets)
        labels = torch.full(
            (batch_size, max_targets), self.num_classes, dtype=torch.int32, device=device
        )
        boxes_cxcywh = torch.zeros((batch_size, max_targets, 4), dtype=dtype, device=device)
        valid_target_mask = torch.zeros((batch_size, max_targets), dtype=torch.bool, device=device)
        for image_index, target in enumerate(targets):
            count = target_counts[image_index]
            if count:
                labels[image_index, :count] = target.labels
                boxes_cxcywh[image_index, :count] = target.boxes
                valid_target_mask[image_index, :count] = True
        labels = labels.tile((1, 2 * num_groups))
        boxes_cxcywh = boxes_cxcywh.tile((1, 2 * num_groups, 1))
        valid_target_mask = valid_target_mask.tile((1, 2 * num_groups))
        negative_indicator = torch.zeros((batch_size, max_targets * 2, 1), device=device)
        negative_indicator[:, max_targets:] = 1
        negative_indicator = negative_indicator.tile((1, num_groups, 1))
        positive_mask = (1 - negative_indicator).squeeze(-1) * valid_target_mask
        flat_positive_indices = torch.nonzero(positive_mask)[:, 1]
        positive_indices = tuple(
            torch.split(flat_positive_indices, [count * num_groups for count in target_counts])
        )
        return _PreparedQueries(
            labels=labels,
            boxes_cxcywh=boxes_cxcywh,
            valid_target_mask=valid_target_mask,
            negative_indicator=negative_indicator,
            positive_indices=positive_indices,
        )

    def _apply_label_noise(
        self,
        query_classes: Int32[Tensor, "batch queries"],
        valid_target_mask: Bool[Tensor, "batch queries"],
    ) -> Int32[Tensor, "batch queries"]:
        """Apply label noise to real labels while leaving padding untouched.

        A real slot is replaced when ``U < 0.5 * label_noise_ratio`` and the
        replacement is sampled uniformly from ``[0, num_classes)``.  The
        ``valid_target_mask`` prevents the reserved padding embedding index
        from being turned into a training label.
        """
        if self.config.label_noise_ratio <= 0:
            return query_classes
        label_noise = torch.rand_like(query_classes, dtype=torch.float) < (
            self.config.label_noise_ratio * 0.5
        )
        noisy_labels = torch.randint_like(
            label_noise, 0, self.num_classes, dtype=query_classes.dtype
        )
        return torch.where(label_noise & valid_target_mask, noisy_labels, query_classes)

    def _apply_box_noise(
        self,
        boxes_cxcywh: Float[Tensor, "batch queries 4"],
        negative_indicator: Float[Tensor, "batch queries 1"],
    ) -> Float[Tensor, "batch queries 4"]:
        """Apply signed center-size box perturbation in the upstream order.

        The implementation converts ``(cx, cy, w, h)`` to corners, adds
        ``box_noise_scale * s * r * (w/2, h/2, w/2, h/2)`` with independent
        random signs ``s`` and magnitudes ``r`` per coordinate, clips corners
        to ``[0, 1]``, and converts back. The result is in image-relative
        ``cxcywh`` units, but it is not guaranteed to be a geometrically valid
        box: independently perturbed corners can cross, producing a signed
        (possibly negative) width or height. When noise is enabled,
        :meth:`forward` passes these values to ``inverse_sigmoid``; that helper
        first clamps values to ``[0, 1]``, then lower-bounds the numerator x
        and denominator 1-x by ``eps`` (default ``1e-5``) before taking their
        log ratio. The intermediate box is not validated or reordered here.
        Negative slots use
        ``r = U + 1`` while positive slots use ``r = U``.  When
        ``box_noise_scale == 0``, the original boxes are returned immediately:
        no random numbers are consumed.  The caller deliberately bypasses
        inverse sigmoid in that case for pinned compatibility behavior.
        """
        if self.config.box_noise_scale <= 0:
            return boxes_cxcywh
        noisy_boxes = _box_cxcywh_to_xyxy(boxes_cxcywh)
        difference = (
            torch.tile(boxes_cxcywh[..., 2:] * 0.5, (1, 1, 2)) * self.config.box_noise_scale
        )
        random_sign = torch.randint_like(boxes_cxcywh, 0, 2) * 2.0 - 1.0
        random_part = torch.rand_like(boxes_cxcywh)
        random_part = (random_part + 1.0) * negative_indicator + random_part * (
            1 - negative_indicator
        )
        noisy_boxes += random_sign * random_part * difference
        return _box_xyxy_to_cxcywh(noisy_boxes.clip(min=0.0, max=1.0))

    @staticmethod
    def _attention_mask(
        actual_queries: int,
        num_queries: int,
        max_targets: int,
        num_groups: int,
        device: torch.device,
    ) -> Bool[Tensor, "total_queries total_queries"]:
        """Build the decoder isolation mask for denoising groups.

        The boolean matrix is indexed as ``mask[query_row, key_column]``;
        ``True`` blocks attention.  Within a DN group and from DN to matching
        queries entries are open, while every DN group is blocked from the
        others and matching queries are blocked from all DN slots.  See the
        class docstring for the compact row/column layout.
        """
        total_queries = actual_queries + num_queries
        attention_mask = torch.zeros(
            (total_queries, total_queries), dtype=torch.bool, device=device
        )
        attention_mask[actual_queries:, :actual_queries] = True
        group_size = max_targets * 2
        for group_index in range(num_groups):
            start = group_size * group_index
            end = start + group_size
            attention_mask[start:end, :start] = True
            attention_mask[start:end, end:actual_queries] = True
        return attention_mask
