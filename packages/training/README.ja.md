# Clean RT-DETR Training

[English version](README.md)

`clean-rtdetr-training` は、`clean-rtdetr` の単一プロセス matching と loss、contrastive denoising query、オプションの Lightning アダプターを提供します。Python 3.13 以降が必要です。

```sh
python -m pip install clean-rtdetr-training
```

検証済み範囲は、CPU FP32/FP64 の loss と勾配のパリティ、および epoch 境界で再開できる小規模な単一プロセス CPU FP32 Lightning 統合です。データセット全体の学習、収束、COCO AP、GPU/DDP/AMP、学習時 augmentation、分散正規化は主張しません。既定レシピの合成データはプログラムの配線を確認するもので、本番用データパイプラインではありません。

ソースから開発する場合は次を実行します。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train --help
uv run pytest packages/training/tests
```

CPU 上でパッケージ化された 2 epoch の合成 Lightning 例を実行します。

```sh
uv run clean-rtdetr-train
```

loss の構築では、公開 API は正規化された cxcywh の target と、0 始まりで連続したクラスラベルを受け取ります。

```python
import torch

from clean_rtdetr import DetectionOutput
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion

model_output = DetectionOutput(
    logits=torch.zeros(1, 2, 3, requires_grad=True),
    boxes=torch.full((1, 2, 4), 0.5, requires_grad=True),
)
labels = torch.tensor([1], dtype=torch.int64)
boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
criterion = SetCriterion(matcher=HungarianMatcher())
losses = criterion(model_output, [DetectionTarget(labels=labels, boxes=boxes)])
total_loss = sum(losses.values())
total_loss.backward()
```

Python API は `clean_rtdetr_training` から公開されています。対応する `clean-rtdetr` リリースに依存し、export パッケージはインストールしません。`clean_rtdetr.training` の互換性 shim も提供しません。

Clean RT-DETR は Apache-2.0 ライセンスで、コミット `068dfde65f2667ad6555883c69d73de886518cad` の `lyuwenyu/RT-DETR` を基にしています。ワークスペースのルートには upstream のソース、ライセンス、詳細な来歴が保持されています。
