# Pinned upstream oracle

Source: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

- Commit: `068dfde65f2667ad6555883c69d73de886518cad`
- File: `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py`
- SHA-256: `c2837740cb66f128b72e802463db57ee3653e15d964baa3b9cf58cc27a817588`
- File: `rtdetr_pytorch/src/zoo/rtdetr/matcher.py`
- SHA-256: `7e364fbba500b0ab23e3e33b7430780cd8dcf4738bd3b961118a3b1f8acd020e`
- File: `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py`
- SHA-256: `54e0f25364bf5c52ae60f8f9363dd2df7d9274657d97a881a2757b7805b739df`
- File: `rtdetr_pytorch/configs/rtdetr/include/optimizer.yml`
- SHA-256: `0499a2af76f497cec3174f69ec4888a53597ba05387a9ab2ac7263d240c3d581`
- File: `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py`
- SHA-256: `a33a3afbc955866eb08587fdc77da0ae446153af952f42b721abc981f413444c`
- File: `rtdetr_pytorch/src/zoo/rtdetr/box_ops.py`
- SHA-256: `7f39e132468a32de058786b01256bc2d801b4f28600fba6b0496ed22678a1a1f`
- File: `rtdetr_pytorch/src/data/transforms.py`
- SHA-256: `eccf323d38ee2ab2eddfe18ea064a9fd30801fa926123f9d85fe2529803577c3`
- File: `rtdetr_pytorch/configs/rtdetr/include/dataloader.yml`
- SHA-256: `be594d0ee141b74e47368afff5674b2212b8c3e849163a6b821fb901e58f6d3c`
- License: [Apache-2.0](LICENSE). The source attribution is retained in the file.

The hashes above describe the checked-in files. `box_ops.py` has one added
trailing newline; its upstream bytes have SHA-256
`a670ce02bce95c3b2304a6708c7c609f281237a7bbe8d63bcef56393a80baa4c`.
Its executable source is unchanged. The other inference provenance files are
byte-for-byte copies of the pinned upstream files.

The decoder file is copied without modifications. Tests extract and execute the
original `_get_encoder_input`, `_generate_anchors`, and `_get_decoder_input`
methods through Python's AST, avoiding the upstream registry and optional training
dependencies. Head and projection weights are copied into this oracle before
comparing outputs and gradients. This validates extracted stages, not a complete
detector or an official trained checkpoint.

Inference parity tests similarly extract the unchanged postprocessor and its
upstream box conversion helper. The preprocessing check is a narrow reference for
the documented RGB, bilinear-resize, byte-decoding, and `/255` validation path; it
does not execute the complete legacy torchvision transform registry.

The production geometry, memory projection, and selection modules are adapted
from this upstream implementation. They contain change notices. This directory
is test-only and is not included in the installed Python wheel.

The pinned R50vd configuration and deformable-attention utility used by the
checkpoint parity oracle are retained under `rtdetr_pytorch/`. The official
checkpoint artifact is kept outside version control in `checkpoints/`; its URL,
SHA-256, and selected EMA state are recorded in `benchmarks/baseline.json`.

The vendored training files retain upstream semantics. `HungarianMatcher`
combines classification, L1 box, and generalized-IoU costs with configurable
focal matching; `SetCriterion` provides cross-entropy, BCE, focal, VFL, box,
mask, auxiliary, and denoising losses. The baseline optimizer configuration
uses AdamW (base LR `1e-4`, backbone LR `1e-5`, weight decay `1e-4`), 72 epochs,
gradient clipping at `0.1`, EMA decay `0.9999`, and a `MultiStepLR` milestone at
1000. These files are provenance references and do not claim validated
training support in this project.

The production `training` package adapts only the focal-cost Hungarian matcher
and Varifocal/L1/GIoU losses, including independent auxiliary matching. Its CPU
tests extract unchanged upstream methods and geometry and compare assignment,
values, and gradients under single-process normalization. Denoising losses and
full training are not implemented. The defaults come from the retained
`rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml`.
