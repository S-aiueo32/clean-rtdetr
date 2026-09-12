# 拡張境界

[English version](../extensions.md)

アーキテクチャや training recipe を変更するときに、パッケージ境界と tensor contract を
保つための拡張点を説明する。ここで扱うのは対応する構築・観測の境界であり、独自の
model、checkpoint、export を検証するものではない。

## 名前付きモデルプリセット

`create_model(name, num_classes=80)` で `list_models()` が返す PResNet variant を
構築する。RT-DETR と RT-DETRv2 のそれぞれに R18vd、R34vd、R50vd-M、R50vd、
R101vd がある。より具体的な戻り値型が必要な場合は、名前付き constructor も
利用できる。

```python
from clean_rtdetr import create_model, get_model_preset

model = create_model("rtdetr_r101vd", num_classes=20)
preset = get_model_preset("rtdetr_r101vd")
print(preset.config_url, preset.checkpoint_url)
```

`ModelPreset` は解決済みのアーキテクチャと upstream の来歴を記録する。URL を
参照してもダウンロードは実行しない。公式変換と parity が検証済みなのは
`rtdetr_r50vd` だけである。他の constructor はアーキテクチャと state-dict 構造を
検査する。

v2 preset は、名前付き encoder projection、selector projection、point-scale state、
identity の FFN residual 処理を共通 stage へ注入する。medium preset は既定の推論用
`DecoderPlan` も注入する。学習時と明示的な深度では要求された decoder prefix を
実行する。モデル実行中に v1/v2 family を切り替える分岐はない。

`QuerySelector` は通常の `nn.Module` インスタンスを受け取る。注入した
モジュールは子モジュールとして登録され、初期化を保持し、autograd に参加し、
`state_dict()` と `parameters()` に現れる。

```python
from torch import nn
from clean_rtdetr import QuerySelector, QuerySelectorConfig

selector = QuerySelector(
    QuerySelectorConfig(hidden_dim=256, num_classes=80, num_queries=300),
    score_head=nn.Linear(256, 80),
    box_head=nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4)),
)
```

score head は `[B,S,K]` の logits を、box head は `[B,S,4]` の座標 delta を生成する。
互換性のない出力 shape はエラーになる。この例は研究用のアーキテクチャ変更であり、
デフォルトで parity を保つ置き換えではない。

## 共有する契約と geometry

共有するテンソルのデータ構造と検証は `clean_rtdetr.contracts`、
anchor 生成と座標変換は `clean_rtdetr.geometry` に置く。
モデルの各ステージの実装と interface は `clean_rtdetr.models` に置く。

```python
from clean_rtdetr.contracts import DecoderPlan, EncodedFeatures
from clean_rtdetr.geometry import generate_anchors, inverse_sigmoid
```

削除済みの `clean_rtdetr.core` サブパッケージではなく、上記のパスを使う。
`from clean_rtdetr import DecoderPlan` など、パッケージ直下からの import は維持する。
tensor の state-dict key と計算処理は変わらない。

## 型パラメータと型エイリアス

ジェネリックなクラス・関数と型エイリアスには
[PEP 695](https://peps.python.org/pep-0695/) の構文を使う。
ワークスペースは Python 3.13 以降を必要とし、Ruff も Python 3.13 を対象にこの規約を検査する。
型パラメータは `class TrainingModel[PredictionT]` のように各宣言のスコープ内で定義し、
`DetectorT: Detector` のようなコンポーネントの上限型を保持する。変性は型チェッカーに推論させる。

jaxtyping のテンソル契約も含め、型エイリアスは `type` 文で宣言する。
`BaselineRTDETR` は型注釈用のエイリアスである。モデルの生成には `RTDETR(...)` または
`rtdetr_r50vd()` を使う。型エイリアスは遅延評価され、実行時のクラスやコンストラクタにはならない。
`ImageInput` は静的検査では Pillow とテンソルの union、実行時には `object` のエイリアスを保つ。
これにより Pillow を任意依存のままにし、明示的な入力検証による既存のエラーも維持する。

<a id="tensor-annotation-conventions"></a>

## Tensor のアノテーション規約

公開境界では `torch.Tensor` とともに
[jaxtyping の array annotations](https://docs.kidger.site/jaxtyping/api/array/) を使う。

- 投影済み memory には `Float[Tensor, "batch sequence channels"]`。
- 正規化済み box 座標／logits には `Float[Tensor, "batch queries 4"]`。
- 選択 index には `Int64[Tensor, "batch queries"]`。
- anchor の有効性（`True=valid`）には `Bool[Tensor, "1 sequence 1"]`。

複数の浮動小数点精度を受け付ける境界では `Float32` ではなく `Float` を使う。
geometry の `*shape` は任意の入力 rank を保持し、box head の `*batch` はすべての先行次元を
保持する。feature level には `_channels _height _width` を使う。これらの軸は level
ごとに変わり得るためである。`FeatureSpec` と明示的な validator が対応関係を定義する。
座標規約と detach policy は shape annotation だけでは表現できないため、文章でも記述する。

この統合では、自動 runtime decorator を追加せずに annotation を導入する。ty は基礎となる
Python 型を検査し、既存の `validate()` メソッドが runtime contract を検証する。記号軸名の
一致だけでは runtime check は生成されない。[upstream FAQ](https://docs.kidger.site/jaxtyping/faq/)
では static checker の制約と Ruff の shape-string parsing を説明している。これらの文字列を
使うモジュールに限り `F722` を除外する。単一の名前付き軸では、先頭に空白を入れた
`" levels"` によって、undefined-name check を無効にせず誤った `F821` を避ける。

annotation は内部テンソルの検証方針も定義する。jaxtyping の契約で dtype と shape が
すでに明示されているテンソルを、内部の各 stage で繰り返し検証しない。公開境界の
明示的な `validate()` は維持し、annotation では表現できない値域やテンソル間の関係も
検証する。内部 helper では、計算前に metadata、device、detach 状態、設定に依存する
関係を確認する必要がある場合に限り、より狭い構造検証を使える。
`MultiScaleFeatures`、`EncodedFeatures`、`DecoderInput` では `validate_structure()` を使い、
外部から受け取る payload の完全な検証には `validate()` を使う。`Float` だけでは
複数テンソルの浮動小数点精度の一致を表さないため、その一致確認は残する。信頼されたモデル
stage 間で生成されるテンソルについて、rank、dtype、記号 shape の重複検証は不要である。
公開 validation の動作と stage の契約は、pytest のみで有効にする jaxtyping の runtime
検査とともにテストで確認する。今回の整理対象は core のモデル stage である。training と
export の境界にある prediction／target 検証は維持する。

<a id="gradient-ownership"></a>

## 勾配の所有権

この境界では `QuerySet.ref_logits` は引き続き勾配を持つ。`QueryAssembler` は
optional な DN reference logits を連結し、decode 前に結果を detach する。
encoder auxiliary box prediction は固有の graph を保持する。テストではこの二つの経路を
明確に区別する。

現在は 4 つの独立した recipe が、分類損失の差し替え、分布回帰 decoder、query 分岐、
AIFI 蒸留を公開 interface から利用している。そこから追加したライブラリ API と、
ユーザー側で所有するアルゴリズム・状態・検証の範囲は
[recipe の責務分担](recipe-boundaries.md) を参照する。
[設計提案](design-proposal.md) はより広い設計案を記録しており、各手法の対応表ではない。
