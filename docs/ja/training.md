# マッチングと学習損失

[English version](../training.md)

`clean_rtdetr_training` は、FP32/FP64 PyTorch 予測に対する固定 R50vd criterion を実装します。内容は Hungarian 割り当て、Varifocal 分類、正規化 cxcywh L1 損失、GIoU 損失、contrastive denoising の構築と損失です。別の [Lightning アダプター](lightning.md) はベースラインの optimizer 設定と単一プロセス CPU FP32/`bf16-mixed` または CUDA FP32/AMP 学習、epoch 境界での再開を実装します。Hydra の config group はベースライン部品と optimizer/scheduler factory を注入します。組み込みモデルは autocast 中も予測値を FP32 に保つため、loss 前の外部キャストは不要です。PyTorch DDP では全 rank の target 数を合計し、world size で割って最小 1 に clamp して正規化します。学習 augmentation は対象外です。研究手法の組み合わせは checkout 専用の [recipe](../../recipes/README.md) に置きます。収束や学習済み精度は保証しません。

<a id="installation-and-cpu-example"></a>

## インストールと CPU 例

```sh
uv sync --package clean-rtdetr-training
uv run python tools/training_losses.py
```

SciPy は上流で使われる `scipy.optimize.linear_sum_assignment` を提供します。これは空でない target のマッチング時だけ import され、推論とパッケージ import には不要です。PyTorch が損失と勾配を計算します。例はデータセットやチェックポイントなしで、CPU の合成 backward pass を実行します。

<a id="api-and-targets"></a>

## API と target

実装モジュールは責務ごとに分かれています。

```text
packages/training/src/clean_rtdetr_training/
  contracts.py          # shared targets, assignments, and validation
  interfaces.py         # shared model, criterion, matcher, and DN interfaces
  losses/
    criterion.py
    supervision.py       # standard matched stages, pairs, and box losses
    matcher.py
    geometry.py
  models/
    denoising.py
    rtdetr.py
  data/
    contracts.py
    synthetic.py
    collate.py
  lightning/
    data.py
    module.py
    optim.py
    ema.py
    state.py
```

`clean_rtdetr_training` の公開 export が組み合わせの入口です。`clean_rtdetr_training.data` と `clean_rtdetr_training.models` も initializer から公開型を提供するため、Hydra target は実装ファイル名に依存しません。Lightning 固有のライフサイクルコードは `clean_rtdetr_training.lightning` にあります。

学習用モデル部品は `models/` に置き、`denoising.py` が DN query を生成し、`rtdetr.py` が detector と generator を組み合わせます。DN の契約 `DenoisingMetadata`、`DenoisingBatch`、`DenoisingOutput` は `clean_rtdetr_training.contracts`、分割 helper は `clean_rtdetr_training.models.rtdetr` に置き、いずれもパッケージ直下から公開します。これらの移動は source API の変更ですが、tensor の state dict key は変えません。

```python
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion, TrainingOutput

# model is an RTDETR instance; images and targets share its device and dtype.
model.train()
output = model(images)  # Default DecoderPlan retains all decoder layers.
targets = [DetectionTarget(labels=labels, boxes=boxes)]  # One target per image.
losses = SetCriterion(matcher=HungarianMatcher())(TrainingOutput(output), targets)
total_loss = sum(losses.values())  # Values are already weighted.
total_loss.backward()
```

各 target は int64 labels `[N]` と浮動小数点 boxes `[N,4]` を持ちます。ラベルは `[0,C)` の連続した 0 始まりクラスインデックスで、**COCO category ID ではありません**。明示的な background class はありません。ボックスは正規化された `(center_x, center_y, width, height)` で `[0,1]`、幅と高さは正です。予測のゼロ面積ボックスは許可され、角座標が画像外に出ても criterion は clip しません。ラベル、ボックス、予測は同じ device を共有し、target box は予測 dtype も共有する必要があります。予測と target は有限値でなければなりません。空画像は labels `[0]`、boxes `[0,4]` です。ignore/crowd 処理と座標変換は学習入力パイプラインの責務です。

`HungarianMatcher()(detection_output, targets)` は画像ごとに 1 つの `MatchIndices` を返します。`prediction_indices` と `target_indices` は int64 で予測 device 上にあります。割り当ては微分不能です。矩形割り当ては `(query 数, target 数)` の小さい方の組数を対応付けます。

`SetCriterion` は `TrainingCriterion[TrainingOutput]` を実装し、module 呼び出しでは
final・auxiliary・任意の DN supervision を持つ `TrainingOutput` を受け取ります。
DN なしで core の `ModelOutput` を学習する場合は `TrainingOutput(model_output)` で包みます。
単一段の `DetectionOutput` の損失は `criterion.stage_losses(detection_output, targets)` で計算します。

core の `ModelOutput` の field は `decoder`、`encoder_aux`、`after_aifi` だけです。decoder と `final` は供給された全 query slot を保持し、準備済み DN query を core に渡した場合は DN prefix も含みます。training wrapper がその prefix を分割して `TrainingOutput(predictions, denoising)` を返します。この `predictions` と `final` は matching query だけを含み、`denoising` が DN slot と割り当てを保持します。

| 項目 | matcher cost weight | 返却損失 weight |
| --- | --- | --- |
| 分類 | 2 (focal cost, alpha 0.25, gamma 2) | 1 (Varifocal, alpha 0.75, gamma 2) |
| L1 ボックス | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` と `CriterionConfig` はこれらの重みと focal パラメータを公開します。`SetCriterion` は割り当て用の抽象 `Matcher` module interface を必要とし、呼び出し元または Hydra 設定が matcher を構築します。custom matcher は Hungarian 固有設定なしに `forward` と `checkpoint_config` を実装できます。基本キーは `loss_vfl`、`loss_bbox`、`loss_giou` です。auxiliary key には `_aux_0`、`_aux_1` などを付け、encoder は上流順序に合わせて最後の auxiliary index を使います。各項は batch 内の総 target 数を `max(..., 1)` で割ります（query より target が多い場合も同じです）。完全に空の batch でも負例の分類損失と微分可能なゼロの box 損失を保持します。Varifocal IoU target と確率重みは detach され、上流の勾配を維持します。local target normalization が不正になるため、複数 rank の分散実行は拒否されます。

`SetCriterion` は標準の VFL、L1、GIoU だけを所有し、分類損失の注入 API は持ちません。既定の設定値と state dict key は維持します。別の分類目的は recipe の criterion で定義し、必要なら `clean_rtdetr_training.losses` の `iter_matched_stages`、`gather_matched_pairs`、`matched_iou_targets`、`matched_box_losses` を再利用します。式と重みは recipe が所有します。stage helper は標準 RT-DETR の matching と DN supervision を保持するため、その方針も変える recipe は stage supervision 自体を実装します。

`iter_matched_stages` は `TrainingOutput` を受け取ります。単一段には
`match_stage(detection_output, targets, matcher)` を使い、返された `MatchedStage` に
同じ損失 helper を適用できます。

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
    rtdetr_r50vd(),
    DenoisingQueryGenerator(80, 256, DenoisingConfig()),
).train()
training_output = training_model(images, targets=targets)
losses = SetCriterion(matcher=HungarianMatcher())(training_output, targets)
sum(losses.values()).backward()
```

wrapper は detector と学習可能な `denoising.class_embed` を一緒に所有し、`TrainingModel[TrainingOutput]` を実装します。matching と DN の両方を学習するには、`TrainingOutput` 全体を `SetCriterion` に渡します。明示的に DN 損失を省く場合は `TrainingOutput(training_output.predictions)` を渡します。任意の統合 interface は [Lightning training](lightning.md#model-and-criterion-interfaces) に記載しています。optimizer 構築には `training_model.parameters()`、重みには `state_dict()` を使います。core detector は dataset target を受け取りません。評価モードで target なしに呼ぶと inference model に委譲し DN noise を生成しません。wrapper の `training_output.final` は matching query だけを含みます。

既定値は上流に合わせて `num_denoising=100`、`label_noise_ratio=0.5`、`box_noise_scale=1.0` です。batch 最大 target 数を `M` とすると、builder は `G=max(1, num_denoising // M)` group、`D=2*M*G` query（positive/negative slot と padding を含む）を使います。要求数は hard cap ではありません。matching query は DN slot に attend できず、DN group 同士も attend できません。ラベル置換確率は `label_noise_ratio / 2` です。box perturbation、clip、RNG 順序、初期 reference の detach は固定された実装に従います。特に `box_noise_scale=0` では上流が inverse sigmoid を省略し、この edge case も保持します。

`generator(targets, num_queries=Q)` は準備済み query と metadata の `DenoisingBatch` を返します。`Detector.forward(..., denoising=prepared.queries)` は `DenoisingQueries` の tensor だけを受け取り、metadata や raw target は受け取らず、noise も生成しません。core は全 decoder slot を分割せずに返し、training wrapper が準備済み query 数を検証して出力を分割します。`num_denoising=0` または全 target が空なら noise を生成せず `None` を返し、wrapper は通常経路を使います。DN 無効時でも batch は空であってはいけません。target と image は embedding の FP32/FP64 dtype と device を共有します。無効・空の処理の前にも既存の target validation が適用されます。

DN 損失キーは保持された各 layer（final を含む）の `loss_vfl_dn_i`、`loss_bbox_dn_i`、`loss_giou_dn_i` です。既知の positive target slot を使い、`max(total targets, 1) * G` で正規化し、Hungarian matching は行いません。negative と padding slot には negative classification supervision を与えます。全 layer の supervision には既定 decoder plan を使います。構築が `None` のとき DN key は返しません。

公式学習重みの互換ローダーは checkout 専用です。元の公式 envelope に明示的な
`source="model"` または `source="ema"` と期待 SHA-256 を指定して、
`tools.compatibility.training.load_official_training_weights(...,
model=training_model, ...)` を使います。パッケージで使う場合は初期化済み
wrapper の native state dict を `torch.save(training_model.state_dict(), path)`
で保存し、`clean_rtdetr.checkpoint.load_model_weights(path,
model=training_model, expected_sha256=...)` でロードします。converter は
推論専用で、training wrapper の checkpoint は作りません。これは重みの
ロードだけなので、再開可能な学習には [Lightning アダプター](lightning.md)
を使います。

<a id="validation-scope"></a>

## 検証範囲

CPU テストは、SHA-256 で固定した上流 matcher/criterion と assignment index、重み付き損失、予測勾配、box geometry を比較します。空・混在 batch、矩形割り当て、auxiliary supervision、不正入力を対象とし、小型モデルの backward で既存モデルとの接続も確認します。これらは完全な学習再現を示しません。DN テストは seed 付き構築、attention mask、metadata、embedding 勾配、各 layer の重み付き損失・予測勾配を固定上流コードと比較します。wrapper テストは登録、空 batch、query 分離、推論動作、合成 backward を確認します。
