# License scope and provenance

[日本語版](ja/licensing.md)

The Apache-2.0 licenses in this repository cover Clean RT-DETR's package and
recipe source code and the Apache-licensed adaptations identified in their
notices. They do not relicense external dependencies, downloaded weights,
datasets, or other separately attributed assets.

| Material | License or applicable terms |
| --- | --- |
| Core, export and training source | Apache-2.0; keep the package LICENSE and bundled provenance notices |
| DEIM, D-FINE, RT-DETRv3 and RT-DETRv4 recipe source | Apache-2.0; keep each recipe's LICENSE, NOTICE.md and source change notices |
| DINOv3 code and pretrained weights | DINOv3 License, supplied separately by Meta |
| Penn-Fudan images and annotations | Dataset provider's terms; no Apache-2.0 grant was found on its distribution page |
| Sample photograph and detection preview | CC0 source photograph; see [sample provenance](../assets/samples/README.md) |
| Other dependencies and downloaded checkpoints | Their own licenses and artifact-specific terms |

## Reference source licenses

The recipe references were checked at these immutable revisions. A source
repository's license is evidence about its code, not a blanket license for
every weight or dataset mentioned by that repository.

| Reference | Pinned source license |
| --- | --- |
| RT-DETR | [068dfde6](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/LICENSE), [29320b6f](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/LICENSE): Apache-2.0 |
| DEIM | [09d35d53](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/LICENSE): Apache-2.0 |
| D-FINE | [956d1709](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/LICENSE): Apache-2.0 |
| RT-DETRv3 | [349e7d99](https://github.com/clxia12/RT-DETRv3/blob/349e7d99a5065e7b684118912e6a74178d4f4625/LICENSE): Apache-2.0 |
| RT-DETRv4 | [55fefaae](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/LICENSE): Apache-2.0 |

Under [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0),
redistribution retains the license text, applicable original notices, and
prominent modification notices in changed files. Preserve any upstream NOTICE
attribution that applies to the reused material. Adding a local copyright
statement does not replace those notices.

The installed core and training packages include provenance under their
`licenses/` directories; the checkout retains the reference sources under
`third_party/RT-DETR`. When copying a recipe independently, copy its LICENSE
and NOTICE.md with its source and retained test fixtures.

## DINOv3 and training artifacts

Meta's [DINOv3 License, August 19, 2025 revision](https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md)
covers DINOv3 code and weights. Section 1(b)(i) requires redistribution of DINO
materials and their derivatives under that agreement with a copy of it;
section 1(b)(ii) requires acknowledgment in research publications. Follow the
terms supplied with the actual checkout and weights; the recipe's Apache
license does not replace them.

The RT-DETRv4 recipe loads a caller-provided local teacher. Although no DINOv3
code or weights are committed here, its training wrapper registers the teacher
as a submodule. A full training state dictionary and Lightning resume checkpoint
therefore include teacher weights; EMA state also contains them when enabled.
A checkpoint made with DINOv3 must not be represented as wholly Apache-2.0.

Extracting just the student detector avoids including the teacher tensors in
that particular file. It does not by itself determine the licensing status of
a distilled model. Keep the teacher revision, weight provenance and applicable
terms with experiment records, and assess the intended artifact before sharing.
The explicit `SmokeTeacher` contains random local test parameters, not DINOv3
materials; using it does not establish pretrained DINOv3 results.

## Datasets and converted weights

The [Penn-Fudan distribution page](https://www.cis.upenn.edu/~jshi/ped_html/)
provides the dataset and attribution but no explicit Apache-2.0 grant. The
loader's code license grants no rights to its images or annotations. Obtain the
dataset separately and resolve permission for the intended use or redistribution
with its provider where the terms are unclear.

Converting a checkpoint's tensor names or exporting its graph does not change
rights in the underlying weights. Record the original provider, artifact and
license separately from the Clean RT-DETR code used to process it.
