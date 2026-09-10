# Clean RT-DETR Training

[English version](README.md)

`clean-rtdetr-training` は、`clean-rtdetr` の単一プロセス matching と loss、contrastive denoising query、オプションの Lightning アダプターを提供します。Python 3.13 以降が必要です。

`SetCriterion` は標準 VFL/L1/GIoU を所有し、旧 `ClassificationLoss` 注入 hook は削除されました。
`HungarianMatcher` は分布回帰向けに `prediction_box_range="unbounded"` を明示的に選択できます。
Lightning adapter は `required_callback_keys` の宣言を checkpoint と再開先へ引き継ぎます。
契約の詳細と、ユーザー側で所有するアルゴリズム・データ・状態は
[recipe の責務分担](../../docs/ja/recipe-boundaries.md)を参照してください。

```sh
python -m pip install clean-rtdetr-training
```

検証済み範囲は、CPU FP32/FP64 の loss と勾配のパリティ、および epoch 境界で再開できる小規模な単一プロセス Lightning 統合です。CUDA FP32 と FP16 AMP、PyTorch DDP の world 平均 target 数による loss 正規化にも対応します。測定済み範囲と制限は [GPU 実行](../../docs/gpu-validation.md) を参照してください。データセット全体の学習、収束、COCO AP、学習時 augmentation は主張しません。サンプルレシピの合成データはプログラムの配線を確認するもので、本番用データパイプラインではありません。

ソースから開発する場合は次を実行します。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train --help
uv run pytest packages/training/tests
```

リポジトリルートから、外部の設定ツリーを使って CPU 上で 2 epoch の合成 Lightning 例を実行します。

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training
```

設定ツリーはリポジトリルートの `configs/` に置き、YAML は training distribution に同梱しません。インストール利用者は独自の設定ツリー、または `configs/` のコピーを指定してください。設定オプションに続けて `trainer.max_epochs=1` などの Hydra override を渡せます。

小規模な実画像ベンチマークとして、`benchmark` extra で Penn-Fudan ローダーを利用できます。既定の分割は学習120枚・評価50枚です。[取得手順と学習例](benchmarks/penn_fudan.md)を参照してください。

`PennFudanDetectionDataset` は `root`、`split`、`seed`、`val_size` を受け取り、
固定の person=0 クラスを使います。dataset factory は model object なしで構築します。
ほかの data factory を使う場合も、クラス数と画像サイズは設定から明示します。

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
losses = criterion.stage_losses(model_output, [DetectionTarget(labels=labels, boxes=boxes)])
total_loss = sum(losses.values())
total_loss.backward()
```

学習出力全体には `criterion(training_output, targets)` を使います。module の入力は
`TrainingOutput` です。DN なしで core の `ModelOutput` を学習する場合は
`TrainingOutput(model_output)` で包みます。

Python API は `clean_rtdetr_training` から公開されています。対応する `clean-rtdetr` リリースに依存し、export パッケージはインストールしません。`clean_rtdetr.training` の互換性 shim も提供しません。

公式 training weight のロードは checkout 専用の
`tools.compatibility.training.load_official_training_weights` で行います。
公式 envelope を training wrapper にロードした後、
`training_model.state_dict()` を `torch.save` で保存します。パッケージで
使う場合は、その native state dict を `clean_rtdetr.checkpoint.load_model_weights` でロードします。
SHA-256 は任意で検証できます。公式重みは
初期化用であり、Lightning の resume checkpoint ではありません。

Clean RT-DETR は Apache-2.0 ライセンスで、コミット `068dfde65f2667ad6555883c69d73de886518cad` の `lyuwenyu/RT-DETR` を基にしています。ワークスペースのルートには upstream のソース、ライセンス、詳細な来歴が保持されています。
