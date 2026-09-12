# COCO バウンディングボックス評価

[English version](../evaluation.md)

予測生成と指標計算は別コマンドです。画像読み込み、前処理、モデル推論、カテゴリ変換、結果 JSON のシリアライズでは pycocotools を import しません。
プライベートな指標アダプター `evaluation/_pycocotools.py` だけが、`evaluate_coco` 呼び出し時にこれを読み込みます。
変更可能な COCO オブジェクトやネイティブ拡張型は公開 API に現れません。

<a id="generate-predictions"></a>

## 予測を生成する

画像サポート付きの core パッケージをインストールし、ローカルの画像、アノテーション、変換済み native state dict checkpoint を使います。公式 envelope のダウンロードと変換は [推論ガイド](inference.md) の `clean-rtdetr-convert` 手順で行ってから、次を実行します。

```sh
python -m pip install "clean-rtdetr[inference]"
clean-rtdetr-coco-predict \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --output outputs/coco-predictions.json
```

予測と `outputs/coco-predictions.report.json` を書き込みます。画像 ID の数値順に 1 枚ずつ読み込み、既存の RGB Pillow 640×640 前処理と 6 層すべてのデコーダを使い、
クリップ、スコア閾値、NMS なしで上位 300 検出を保持します。既定は CPU FP32 で、`--device` で別の PyTorch デバイスを選べます。画像や重みはダウンロードしません。

`--limit N` はスモーク実行用に最初の N 枚を選びます。レポートには実際に選んだ ID、画像・チェックポイント・アノテーションのハッシュ、コードと環境、
全アノテーション画像か部分集合かが記録されます。画像サイズはアノテーションと一致し、画像パスは `--images` 内に残らなければなりません。既存の出力・レポートは拒否されます。

`coco_predictions(detections, image_ids)` はプロジェクト所有の `Detections` を `image_id`、`category_id`、絶対 xywh の `bbox`、`score` を持つ通常のレコードへ変換します。
ラベル 0–79 は連続しない 80 個の COCO カテゴリ ID に対応します。例えばラベル 11 はカテゴリ 13 です。順序は [固定された RT-DETR マッピング](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/data/coco/coco_dataset.py) に従います。
アノテーションのカテゴリ順は無関係ですが、ID はその集合と完全に一致しなければなりません。

<a id="calculate-ap-from-saved-json"></a>

## 保存済み JSON から AP を計算する

この手順だけ evaluation extra が必要です。推論とは別の環境でも実行できます。

```sh
python -m pip install "clean-rtdetr[evaluation]"
clean-rtdetr-coco-evaluate \
  --annotations datasets/coco/annotations/instances_val2017.json \
  --predictions outputs/coco-predictions.json \
  --prediction-report outputs/coco-predictions.report.json \
  --report outputs/coco-metrics.json
```

`evaluation` extra は pycocotools と推移的な NumPy 依存関係を追加します。uv 設定はソースからの pycocotools ビルドを許可しないため、未対応環境では C コンパイラを呼び出さず、この extra のインストールに失敗します。予測生成は inference extra だけで利用できます。

オプションの予測レポートは両方の入力 JSON とハッシュ照合され、検出のない画像を含む評価対象画像 ID の完全な集合を提供します。レポートがなければ **すべてのアノテーション画像**を評価し、ID を予測から推測することはありません。そのため部分的な予測ファイルが AP から未処理画像を黙って除外することはありません。

`evaluate_coco(annotations, predictions, image_ids=...)` はパス、通常の予測レコード、オプションの画像 ID だけを受け付け、Python の float または `None` の辞書を返します。公式 bbox evaluator は標準の IoU・recall 閾値、crowd/ignore 処理、面積範囲、maxDets `[1,10,100]` を保持します。最後の設定はモデルの top-300 後処理とは異なります。[公式評価実装](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) を参照してください。

レポートの AP と AR は **0–100 単位**です。正解データがないサイズ範囲など、COCO の未定義値は JSON の `null` になります。空の予測リストも利用でき、評価器が呼び出し元のレコードを変更することはありません。

<a id="validation-scope"></a>

## 検証範囲

オフラインテストは小さな合成アノテーションで、カテゴリ変換、完全・欠落・空の予測、画像部分集合、無視される crowd アノテーション、オプション依存分離を確認します。これらは評価経路を確立するもので、モデルの COCO 精度を示すものではありません。別途、公式学習済み R50vd EMA 重みの CPU FP32 推論で、COCO val2017 全 5,000 枚の AP 53.05、AP50 71.22、AP75 57.67 を測定しました。[結果・再現手順](../../benchmarks/coco-val2017.md)と、環境・ソース・入力のハッシュおよび全画像の網羅を記録した[来歴レポート](../../benchmarks/coco-val2017.json)を参照してください。同一条件での上流比較、GPU・レイテンシ測定は未実施です。部分集合のスモーク結果を COCO 全体の AP として報告してはいけません。
