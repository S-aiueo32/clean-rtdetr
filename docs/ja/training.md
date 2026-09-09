# マッチングと学習損失

[English version](../training.md)

`clean_rtdetr_training` は、単一プロセスの FP32/FP64 PyTorch 予測に対する固定 R50vd criterion を実装します。内容は Hungarian 割り当て、Varifocal 分類、正規化 cxcywh L1 損失、GIoU 損失、contrastive denoising の構築と損失です。別の [Lightning アダプター](lightning.md) はベースラインの optimizer 設定と単一プロセス CPU FP32 学習、epoch 境界での再開を実装します。Hydra の config group はベースライン部品と optimizer/scheduler factory を注入します。学習 augmentation、分散正規化、mixed precision、研究手法の組み合わせは未実装です。収束や学習済み精度は保証しません。

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
    matcher.py
    geometry.py
  denoising/
    queries.py
    model.py
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

`clean_rtdetr_training` の公開 export が組み合わせの入口です。`clean_rtdetr_training.data` と `clean_rtdetr_training.denoising` も initializer から公開型を提供するため、Hydra target は実装ファイル名に依存しません。Lightning 固有のライフサイクルコードは `clean_rtdetr_training.lightning` にあります。

```python
from clean_rtdetr_training import DetectionTarget, HungarianMatcher, SetCriterion

# model is an RTDETR instance; images and targets share its device and dtype.
model.train()
output = model(images)  # Default DecoderPlan retains all decoder layers.
targets = [DetectionTarget(labels=labels, boxes=boxes)]  # One target per image.
losses = SetCriterion(matcher=HungarianMatcher())(output, targets)
total_loss = sum(losses.values())  # Values are already weighted.
total_loss.backward()
```

各 target は int64 labels `[N]` と浮動小数点 boxes `[N,4]` を持ちます。ラベルは `[0,C)` の連続した 0 始まりクラスインデックスで、**COCO category ID ではありません**。明示的な background class はありません。ボックスは正規化された `(center_x, center_y, width, height)` で `[0,1]`、幅と高さは正です。予測のゼロ面積ボックスは許可され、角座標が画像外に出ても criterion は clip しません。ラベル、ボックス、予測は同じ device を共有し、target box は予測 dtype も共有する必要があります。予測と target は有限値でなければなりません。空画像は labels `[0]`、boxes `[0,4]` です。ignore/crowd 処理と座標変換は学習入力パイプラインの責務です。

`HungarianMatcher()(detection_output, targets)` は画像ごとに 1 つの `MatchIndices` を返します。`prediction_indices` と `target_indices` は int64 で予測 device 上にあります。割り当ては微分不能です。矩形割り当ては `(query 数, target 数)` の小さい方の組数を対応付けます。

`SetCriterion` は 1 stage の `DetectionOutput`、または final と auxiliary supervision を含む `ModelOutput` を受け取ります。後者では保持された中間 decoder 予測の後に encoder auxiliary prediction が続き、各 stage を独立にマッチングします。全 decoder 損失には既定の `DecoderPlan(collect_intermediate=True)` を使ってください。破棄した中間予測は supervision できません。主 decoder field は matching query だけを含み、任意の `denoising` field は既知の target 割り当てを持つ DN 予測を別に保持します。

| 項目 | matcher cost weight | 返却損失 weight |
| --- | --- | --- |
| 分類 | 2 (focal cost, alpha 0.25, gamma 2) | 1 (Varifocal, alpha 0.75, gamma 2) |
| L1 ボックス | 5 | 5 |
| GIoU | 2 | 2 |

`MatcherConfig` と `CriterionConfig` はこれらの重みと focal パラメータを公開します。`SetCriterion` は割り当て用の抽象 `Matcher` module interface を必要とし、呼び出し元または Hydra 設定が matcher を構築します。custom matcher は Hungarian 固有設定なしに `forward` と `checkpoint_config` を実装できます。基本キーは `loss_vfl`、`loss_bbox`、`loss_giou` です。auxiliary key には `_aux_0`、`_aux_1` などを付け、encoder は上流順序に合わせて最後の auxiliary index を使います。各項は batch 内の総 target 数を `max(..., 1)` で割ります（query より target が多い場合も同じです）。完全に空の batch でも負の分類損失と微分可能なゼロの box 損失を保持します。Varifocal IoU target と確率重みは detach され、上流の勾配を維持します。local target normalization が不正になるため、複数 rank の分散実行は拒否されます。

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
output = training_model(images, targets=targets)
losses = SetCriterion(matcher=HungarianMatcher())(output, targets)
sum(losses.values()).backward()
```

wrapper は detector と学習可能な `denoising.class_embed` を一緒に所有します。`TrainingModel[ModelOutput]` を実装し、`SetCriterion` は `TrainingCriterion[DetectionOutput | ModelOutput]` を実装します。任意の統合 interface は [Lightning training](lightning.md#model-and-criterion-interfaces) に記載しています。optimizer 構築には `training_model.parameters()`、重みには `state_dict()` を使います。core detector は dataset target を受け取りません。評価モードで target なしに呼ぶと inference model に委譲し DN noise を生成しません。`output.final` は常に matching query だけです。

既定値は上流に合わせて `num_denoising=100`、`label_noise_ratio=0.5`、`box_noise_scale=1.0` です。batch 最大 target 数を `M` とすると、builder は `G=max(1, num_denoising // M)` group、`D=2*M*G` query（positive/negative slot と padding を含む）を使います。要求数は hard cap ではありません。matching query は DN slot に attend できず、DN group 同士も attend できません。ラベル置換確率は `label_noise_ratio / 2` です。box perturbation、clip、RNG 順序、初期 reference の detach は固定された実装に従います。特に `box_noise_scale=0` では上流が inverse sigmoid を省略し、この edge case も保持します。

`generator(targets, num_queries=Q)` は準備済み query と metadata の `DenoisingBatch` を返します。`num_denoising=0` または全 target が空なら noise を生成せず `None` を返し、wrapper は通常経路を使います。DN 無効時でも batch は空であってはいけません。target と image は embedding の FP32/FP64 dtype と device を共有します。無効・空の処理の前にも既存の target validation が適用されます。

DN 損失キーは保持された各 layer（final を含む）の `loss_vfl_dn_i`、`loss_bbox_dn_i`、`loss_giou_dn_i` です。既知の positive target slot を使い、`max(total targets, 1) * G` で正規化し、Hungarian matching は行いません。negative と padding slot には negative classification supervision を与えます。全 layer の supervision には既定 decoder plan を使います。構築が `None` のとき DN key は返しません。

公式互換ローダーは checkout 専用です。元の公式 envelope に明示的な
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
