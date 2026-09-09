# 最初の拡張境界

[English version](../extensions.md)

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

<a id="gradient-ownership"></a>

## 勾配の所有権

この境界では `QuerySet.ref_logits` は引き続き勾配を持ちます。将来の query assembler は
optional な DN reference logits を連結し、decode 前に結果を detach する必要があります。
encoder auxiliary box prediction は固有の graph を保持します。テストではこの二つの経路を
明確に区別します。

これは意図的に範囲を絞った最初の境界です。D-FINE の distribution state、query group、
AIFI feature tap、training method、capability validation は
[設計提案](design-proposal.md) で計画されています。placeholder API でこれらを
シミュレートすることはありません。
