# COCO bounding-box 評価

[English version](../evaluation.md)

これらのコマンドで、ローカル画像から COCO 予測 JSON を作成し、バウンディングボックスの
評価指標を計算する。画像の予測に pycocotools が不要になるよう、予測生成と指標の計算は分けている。

## 前提条件

Python 3.13 以降、変換済み native R50vd チェックポイント、COCO 画像ディレクトリ、その
アノテーション JSON が必要である。公式重みは [推論ガイド](inference.md) の手順で変換する。
どちらのコマンドも画像と重みをダウンロードしない。

予測生成には Pillow が必要である。

```sh
python -m pip install "clean-rtdetr[inference]"
```

評価指標の計算には evaluation extra が必要で、予測とは別の環境でも実行できる。

```sh
python -m pip install "clean-rtdetr[evaluation]"
```

この extra は pycocotools と NumPy を提供する。チェックアウトでは uv の設定により
pycocotools をソースからビルドしないため、`uv sync --package clean-rtdetr --extra evaluation` は
未対応の環境で C コンパイラを起動せず失敗する。この制限は上記の `pip` コマンドには適用されない。

<a id="generate-predictions"></a>

## 予測 JSON を生成する

COCO 画像ファイルのディレクトリと対応するアノテーション JSON を指定する。

```sh
clean-rtdetr-coco-predict \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --output outputs/coco-predictions.json
```

予測 JSON と `outputs/coco-predictions.report.json` を書き出す。画像は数値の image-ID 順に 1 枚ずつ
処理し、Pillow による RGB への 640×640 前処理、6 個の decoder layer、top-300 の検出後処理を使う。
切り詰め、NMS、スコアしきい値は適用しない。既定値は CPU FP32 で、`--device` で PyTorch のデバイスを
選べる。

`--limit N` はスモーク実行のみに使う。レポートは選択した image ID、入力ハッシュ、実行環境、
アノテーション画像全体か部分集合かを記録する。画像寸法はアノテーションと一致し、ファイル名は
`--images` 配下になければならない。既存の出力先とレポート先は拒否される。

`coco_predictions(detections, image_ids)` はプロジェクトの `Detections` を、`image_id`、`category_id`、
絶対座標 `xywh` の `bbox`、`score` を持つレコードに変換する。label 0–79 は連番ではない 80 個の
COCO ID に対応し、たとえば label 11 は category 13 である。アノテーションの category の順序は問わないが、
ID の集合は完全に一致しなければならない。対応表は [固定した RT-DETR
ソース](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/data/coco/coco_dataset.py)
に従う。

<a id="calculate-ap-from-saved-json"></a>

## 保存済み予測から評価指標を計算する

予測実行で `--limit` を使った場合など、部分集合を選択したときは予測レポートを渡す。

```sh
clean-rtdetr-coco-evaluate \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --predictions outputs/coco-predictions.json \
  --prediction-report outputs/coco-predictions.report.json \
  --report outputs/coco-metrics.json
```

レポートのハッシュは両方の入力 JSON と照合され、検出がない画像を含む評価対象の image-ID 全体を指定する。
`--prediction-report` がなければアノテーションの全画像を評価する。評価器は予測レコードから ID を
推測しない。そのため、部分的な予測ファイルが見逃し画像を黙って除外することはない。

`evaluate_coco(annotations, predictions, image_ids=...)` はアノテーションのパス、通常の予測レコード、
任意の ID を受け取る。Python の float または `None` を返し、呼び出し元のレコードを変更しない。
標準 COCO bbox 評価器の IoU/recall しきい値、crowd/ignore 処理、面積範囲、
`maxDets=[1,10,100]` を使う。これはモデル側の top-300 候補とは別である。JSON レポートの評価指標は
0–100 の尺度で、COCO の未定義値は `null` になる。空の予測リストも有効である。

<a id="validation-scope"></a>

## 測定済みの範囲

テストは小さい合成アノテーションを使い、変換、欠損・空の予測、部分集合、crowd アノテーション、
optional dependency の分離を検査する。モデルの精度を確立するものではない。公式学習済み R50vd EMA
重みを使った別の CPU FP32 全件実行では、COCO val2017 全 5,000 枚で AP 53.05、AP50 71.22、AP75 57.67 を
測定した。[結果](../../benchmarks/coco-val2017.md) と
[来歴レポート](../../benchmarks/coco-val2017.json)を参照する。スモーク実行の部分集合は
full COCO AP ではない。同一条件の upstream 比較と GPU/latency 測定は未実施である。
