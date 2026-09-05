# Copyright 2026 Clean RT-DETR contributors
"""Run the implemented feature -> memory -> query -> decoder path with random weights."""

import torch

from clean_rtdetr import (
    Decoder,
    DecoderConfig,
    DecoderPlan,
    FeatureSpec,
    MemoryProjector,
    MultiScaleFeatures,
    QueryAssembler,
    QuerySelector,
    QuerySelectorConfig,
)

torch.manual_seed(0)
specs = (FeatureSpec(32, 8), FeatureSpec(64, 16), FeatureSpec(128, 32))
features = MultiScaleFeatures(
    (torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8), torch.randn(2, 128, 4, 4)),
    specs,
)
projector = MemoryProjector(specs, hidden_dim=32).eval()
selector = QuerySelector(QuerySelectorConfig(hidden_dim=32, num_queries=20)).eval()
assembler = QueryAssembler()
decoder = Decoder(
    DecoderConfig(hidden_dim=32, num_layers=2, num_levels=len(specs)),
).eval()
with torch.inference_mode():
    memory = projector(features)
    selected = selector(memory)
    decoded = decoder(
        assembler(selected.queries),
        memory,
        plan=DecoderPlan(collect_intermediate=False),
    )
print("memory:", tuple(memory.memory.shape))
print("query content:", tuple(selected.queries.content.shape))
print("encoder auxiliary boxes:", tuple(selected.encoder_boxes.shape))
print("decoder logits:", tuple(decoded.final.logits.shape))
print("decoder boxes:", tuple(decoded.final.boxes.shape))
