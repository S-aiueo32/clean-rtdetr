# マッチングと学習損失

[English](../training.md) | 日本語

`clean_rtdetr_training` は、固定した R50vd の学習 criterion を提供する。
Hungarian マッチング、Varifocal 分類、正規化 cxcywh の L1・GIoU 損失、
contrastive denoising（DN）の query と損失を含む。
PyTorch の学習ループや recipe 固有の criterion を組み立てる際に使う。
予測とターゲットは FP32/FP64 で検証し、画像拡張と学習データの変換は呼び出し側が担当する。

criterion は native PyTorch の分散実行に対応する。
全 rank のターゲット数を合計し、world size で割り、下限を 1 として損失を正規化する。
これは DDP の勾配平均と整合する。全 rank で同じ criterion 呼び出しが必要である。
別の [Lightning アダプター](lightning.md)は複数 rank に対応しない。
収束や学習後の精度は実証していない。

<a id="installation-and-cpu-example"></a>

## インストールして動作を確認する

Python 3.13 または 3.14 と uv を使い、リポジトリのルートから実行する。

```sh
uv sync --package clean-rtdetr-training
uv run python tools/training_losses.py
```

このコマンドは合成入力で CPU の逆伝播を実行し、データセットやチェックポイントは不要である。
SciPy の `scipy.optimize.linear_sum_assignment` は、空でないターゲットのマッチング時に使う。
パッケージの import と推論では SciPy を import しない。

<a id="api-and-targets"></a>

## 入力・出力・損失

構成に使う公開型は `clean_rtdetr_training` から import できる。
`clean_rtdetr_training.data`、`.models`、`.lightning` も、直接利用や Hydra による構築に使える公開型を提供する。

```python
import torch

from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion, TrainingOutput

model = rtdetr_r50vd().train()
images = torch.rand(1, 3, 128, 128)
labels = torch.tensor([0])
boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
output = model(images)  # 既定の DecoderPlan はすべての decoder layer を保持する。
targets = [DetectionTarget(labels=labels, boxes=boxes)]
losses = SetCriterion(matcher=HungarianMatcher())(TrainingOutput(output), targets)
sum(losses.values()).backward()  # 返却値にはすでに weight が適用されている。
```

画像ごとに 1 つの `DetectionTarget` を渡す。`labels` は int64 `[N]` で、
クラス ID は `[0, C)` の連続した 0 始まりの値である。COCO の category ID ではなく、背景クラスもない。
`boxes` は FP32 または FP64 の `[N, 4]` である。
正規化した `(center_x, center_y, width, height)` は `[0, 1]` 内、幅と高さは正とする。
ターゲットと予測は有限値で同じ device 上に置き、ターゲットの box は予測と dtype も揃える。
物体のない画像は labels `[0]`、boxes `[0, 4]` で表す。
予測の box は面積 0 を許可し、角座標が画像の外に出ても clip しない。
座標変換と crowd/ignore アノテーションの処理は、この API を呼ぶ前に行う。

`HungarianMatcher()(detection_output, targets)` は、画像ごとに 1 つの `MatchIndices` を返す。
予測・ターゲットの index は int64 で、予測と同じ device にある。
割り当ては微分不能で、最大 `min(queries, targets)` 組を対応付ける。

`SetCriterion` は、最終・補助予測と任意の DN 教師情報を持つ `TrainingOutput` を受け取る。
DN がない core の `ModelOutput` は `TrainingOutput(output)` で包む。
単一の `DetectionOutput` には `criterion.stage_losses(output, targets)` を使う。

| 項目 | マッチングのコスト係数 | 返却損失の重み |
| --- | --- | --- |
| 分類 | 2、focal: alpha 0.25, gamma 2 | 1、Varifocal: alpha 0.75, gamma 2 |
| L1 box | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` と `CriterionConfig` で、これらの重みと focal パラメータを設定できる。
基本キーは `loss_vfl`、`loss_bbox`、`loss_giou` である。
decoder の補助損失には `_aux_0`、`_aux_1` などを付け、encoder は最後の補助 index を使う。
各項は冒頭で説明した全 rank の平均ターゲット数で正規化する。
全画像のターゲットが空でも、負例の分類損失と微分可能なゼロの box 損失を保持する。
Varifocal の IoU ターゲットと確率による重みは detach する。

標準 criterion は VFL・L1・GIoU を所有する。
分類式、マッチング方針、DN の教師情報を変更する recipe は、その方針を自ら定義する。
必要に応じて `clean_rtdetr_training.losses` の `iter_matched_stages`、
`gather_matched_pairs`、`matched_iou_targets`、`matched_box_losses` を再利用できる。
独自の matcher は `forward` と `checkpoint_config` を実装する。

<a id="contrastive-denoising"></a>

## Contrastive denoising

```python
from clean_rtdetr import rtdetr_r50vd
from clean_rtdetr_training import (
    DenoisingConfig,
    DenoisingQueryGenerator,
    DenoisingTrainingModel,
    HungarianMatcher,
    SetCriterion,
)

training_model = DenoisingTrainingModel(
    rtdetr_r50vd(), DenoisingQueryGenerator(80, 256, DenoisingConfig())
).train()
# 前の例で作った images と targets を再利用する。
training_output = training_model(images, targets=targets)
losses = SetCriterion(matcher=HungarianMatcher())(training_output, targets)
sum(losses.values()).backward()
```

`DenoisingTrainingModel` は検出器と学習可能な `denoising.class_embed` を所有する。
optimizer には wrapper の `parameters()`、重みには `state_dict()` を使う。
core の検出器には、データセットのターゲットを直接渡さず、準備済み query テンソルを渡す。
wrapper が先頭の DN 部分を分離し、`TrainingOutput.predictions` と `.final` には
matching query の予測、`.denoising` には DN 予測と metadata を保持する。
両方を学習する場合は出力全体を渡し、DN 損失を省く場合は
`TrainingOutput(training_output.predictions)` を渡す。
評価モードでターゲットなしに呼ぶと、通常の推論に委譲する。

既定値は `num_denoising=100`、`label_noise_ratio=0.5`、`box_noise_scale=1.0` である。
バッチ内の最大ターゲット数を `M` とすると、DN は `G = max(1, num_denoising // M)` 個の
group と `2 * M * G` 個の slot を作る。要求数は上限ではない。
`num_denoising=0` または全ターゲットが空の場合は DN batch を返さない。
保持した各 decoder 層の DN 損失キーは `loss_vfl_dn_i`、`loss_bbox_dn_i`、`loss_giou_dn_i` である。
Hungarian マッチングは行わず既知の正例割り当てを使い、下限を 1 とした全 rank の平均ターゲット数に
`G` を掛けて正規化する。単一プロセスでは `max(total targets, 1) * G` である。

`tools.compatibility.training.load_official_training_weights` は checkout 専用で、
公式形式の重みから wrapper 全体を初期化する。パッケージを使う場合は
`torch.save(training_model.state_dict(), path)` で初期化済み wrapper を保存し、再構築した wrapper に
`clean_rtdetr.checkpoint.load_model_weights(path, model=training_model)` で読み込む。
推論用の変換コマンドは学習チェックポイントを生成しない。
optimizer・scheduler・EMA・エポック境界からの再開状態が必要なら
[Lightning による学習](lightning.md#state-and-resume)を使う。

<a id="validation-scope"></a>

## 検証範囲

CPU の一致検証では、固定した上流 matcher/criterion と、割り当て・重み付き損失・予測の勾配を比較する。
空・混在バッチ、予測数とターゲット数が異なる割り当て、補助損失、DN の構築と損失、
wrapper の query 分離、合成入力の逆伝播を対象とする。
2-rank の損失テストでは、ターゲット数が異なる場合と空の場合を確認する。
これらは学習全体、画像拡張、学習後の精度の再現を示すものではない。
