# Equation provenance

This example implements only Matchability-Aware Loss (MAL) from
[DEIM: DETR with Improved Matching for Fast Convergence (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html).

Reference: [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM),
commit `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`,
`engine/deim/deim_criterion.py`, `DEIMCriterion.loss_labels_mal`, with
`mal_alpha=None` and `gamma=1.5`. Reference implementation:
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
Modified from D-FINE, Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
Upstream is Apache-2.0 licensed. This recipe expresses the equation through
Clean RT-DETR's existing public TrainingCriterion interface rather than
depending on upstream registries or private library methods.

For matched class indicator t, detached sigmoid probability p, and detached
matched IoU q (zero elsewhere): y = q^gamma,
w = t + (1-t) p^gamma, loss = sum(w BCEWithLogits(logits, y)) / N.
N is the local target count clamped to one (times group count for denoising).
Baseline matching, box regression, and per-stage assignments remain RT-DETR.

The stage orchestration and box-loss layout adapt Clean RT-DETR's
`SetCriterion` (Copyright 2026 Clean RT-DETR contributors), itself based on
`lyuwenyu/RT-DETR` commit `068dfde65f2667ad6555883c69d73de886518cad`
(Apache-2.0). The copied recipe retains the Apache-2.0 license in `LICENSE`.
