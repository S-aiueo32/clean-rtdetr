# 拡張境界

[English version](../extensions.md)

## 名前付きモデルプリセット

`create_model(name, num_classes=80)` は `list_models()` に含まれる公式 PResNet
variant を構築します。RT-DETR と RT-DETRv2 のそれぞれについて R18vd、R34vd、
R50vd-M、R50vd、R101vd を選べます。静的に具体的な戻り値型が必要な場合は、対応する
名前付き関数も利用できます。

```python
from clean_rtdetr import create_model, get_model_preset

model = create_model("rtdetr_r101vd", num_classes=20)
preset = get_model_preset("rtdetr_r101vd")
print(preset.config_url, preset.checkpoint_url)
```

`ModelPreset` は解決済みのアーキテクチャ値と upstream の来歴を固定します。URL を
参照してもダウンロードは実行しません。公式変換と parity が検証済みなのは従来の
`rtdetr_r50vd` だけです。他の constructor は現在、アーキテクチャと state-dict 構造を
検査しています。

v2 preset は、名前付き encoder projection、selector projection、point-scale state、
identity の FFN residual 処理を共通 stage へ注入します。medium preset は既定の推論用
`DecoderPlan` も注入します。学習時と明示的な深度では要求された decoder prefix を
実行します。モデル実行中に v1/v2 family を切り替える分岐はありません。

`QuerySelector` は通常の `nn.Module` インスタンスを受け取ります。注入した
モジュールは子モジュールとして登録され、初期化を保持し、autograd に参加し、
`state_dict()` と `parameters()` に現れます。

```python
from torch import nn
from clean_rtdetr import QuerySelector, QuerySelectorConfig

selector = QuerySelector(
    QuerySelectorConfig(hidden_dim=256, num_classes=80, num_queries=300),
    score_head=nn.Linear(256, 80),
    box_head=nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4)),
)
```

score head は `[B,S,K]` の logits を、box head は `[B,S,4]` の座標 delta を生成します。
互換性のない出力 shape はエラーになります。この例は研究用のアーキテクチャ変更であり、
デフォルトで parity を保つ置き換えではありません。

## 共有する契約と geometry

共有するテンソルのデータ構造と検証は `clean_rtdetr.contracts`、
anchor 生成と座標変換は `clean_rtdetr.geometry` に置きます。
モデルの各ステージの実装と interface は `clean_rtdetr.models` に置きます。

```python
from clean_rtdetr.contracts import DecoderPlan, EncodedFeatures
from clean_rtdetr.geometry import generate_anchors, inverse_sigmoid
```

`clean_rtdetr.core.contracts` と `clean_rtdetr.core.geometry` からの import は、
上記のパスへ変更してください。`clean_rtdetr.core` サブパッケージは削除しました。
`from clean_rtdetr import DecoderPlan` など、既存のパッケージ直下からの import は維持します。
テンソルの state dict key と計算処理は変わりません。

## 型パラメータと型エイリアス

ジェネリックなクラス・関数と型エイリアスには
[PEP 695](https://peps.python.org/pep-0695/) の構文を使います。
ワークスペースは Python 3.13 以降を必要とし、Ruff も Python 3.13 を対象にこの規約を検査します。
型パラメータは `class TrainingModel[PredictionT]` のように各宣言のスコープ内で定義し、
`DetectorT: Detector` のようなコンポーネントの上限型を保持します。変性は型チェッカーに推論させます。

jaxtyping のテンソル契約も含め、型エイリアスは `type` 文で宣言します。
`BaselineRTDETR` は型注釈用のエイリアスです。モデルの生成には `RTDETR(...)` または
`rtdetr_r50vd()` を使います。型エイリアスは遅延評価され、実行時のクラスやコンストラクタにはなりません。
`ImageInput` は静的検査では Pillow とテンソルの union、実行時には `object` のエイリアスを保ちます。
これにより Pillow を任意依存のままにし、明示的な入力検証による既存のエラーも維持します。

<a id="tensor-annotation-conventions"></a>

## Tensor のアノテーション規約

公開境界では `torch.Tensor` とともに
[jaxtyping の array annotations](https://docs.kidger.site/jaxtyping/api/array/) を使います。

- 投影済み memory には `Float[Tensor, "batch sequence channels"]`。
- 正規化済み box 座標／logits には `Float[Tensor, "batch queries 4"]`。
- 選択 index には `Int64[Tensor, "batch queries"]`。
- anchor の有効性（`True=valid`）には `Bool[Tensor, "1 sequence 1"]`。

複数の浮動小数点精度を受け付ける境界では `Float32` ではなく `Float` を使います。
geometry の `*shape` は任意の入力 rank を保持し、box head の `*batch` はすべての先行次元を
保持します。feature level には `_channels _height _width` を使います。これらの軸は level
ごとに変わり得るためです。`FeatureSpec` と明示的な validator が対応関係を定義します。
座標規約と detach policy は shape annotation だけでは表現できないため、文章でも記述します。

この統合では、自動 runtime decorator を追加せずに annotation を導入します。ty は基礎となる
Python 型を検査し、既存の `validate()` メソッドが runtime contract を検証します。記号軸名の
一致だけでは runtime check は生成されません。[upstream FAQ](https://docs.kidger.site/jaxtyping/faq/)
では static checker の制約と Ruff の shape-string parsing を説明しています。これらの文字列を
使うモジュールに限り `F722` を除外します。単一の名前付き軸では、先頭に空白を入れた
`" levels"` によって、undefined-name check を無効にせず誤った `F821` を避けます。

annotation は内部テンソルの検証方針も定義します。jaxtyping の契約で dtype と shape が
すでに明示されているテンソルを、内部の各 stage で繰り返し検証しません。公開境界の
明示的な `validate()` は維持し、annotation では表現できない値域やテンソル間の関係も
検証します。内部 helper では、計算前に metadata、device、detach 状態、設定に依存する
関係を確認する必要がある場合に限り、より狭い構造検証を使えます。
`MultiScaleFeatures`、`EncodedFeatures`、`DecoderInput` では `validate_structure()` を使い、
外部から受け取る payload の完全な検証には `validate()` を使います。`Float` だけでは
複数テンソルの浮動小数点精度の一致を表さないため、その一致確認は残します。信頼されたモデル
stage 間で生成されるテンソルについて、rank、dtype、記号 shape の重複検証は不要です。
公開 validation の動作と stage の契約は、pytest のみで有効にする jaxtyping の runtime
検査とともにテストで確認します。今回の整理対象は core のモデル stage です。training と
export の境界にある prediction／target 検証は維持します。

<a id="gradient-ownership"></a>

## 勾配の所有権

この境界では `QuerySet.ref_logits` は引き続き勾配を持ちます。`QueryAssembler` は
optional な DN reference logits を連結し、decode 前に結果を detach します。
encoder auxiliary box prediction は固有の graph を保持します。テストではこの二つの経路を
明確に区別します。

現在は 4 つの独立した recipe が、分類損失の差し替え、分布回帰 decoder、query 分岐、
AIFI 蒸留を公開 interface から利用しています。そこから追加したライブラリ API と、
ユーザー側で所有するアルゴリズム・状態・検証の範囲は
[recipe の責務分担](recipe-boundaries.md) を参照してください。
[設計提案](design-proposal.md) はより広い設計案を記録しており、各手法の対応表ではありません。
