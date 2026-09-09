# 推論の入出力

[English version](../inference.md)

core パッケージには checkpoint converter が含まれます。次でインストールします。

```sh
python -m pip install clean-rtdetr
```

Pillow はオプション依存関係です。画像サポートをインストールするには、次を実行します。

```sh
python -m pip install "clean-rtdetr[inference]"
```

checkout の開発環境では `uv sync --package clean-rtdetr --extra inference` を使い、
以下のコマンドに `uv run` を付けます。

ONNX エクスポートと CPU ONNX Runtime 検証器を使う場合は、別の
`clean-rtdetr-export` パッケージを `python -m pip install clean-rtdetr-export`
でインストールします。checkout の開発環境では `uv sync --package
clean-rtdetr-export` を使います。

<a id="model-tensors"></a>

## モデルのテンソル

`RTDETR` は、`[B,3,H,W]` 配置の前処理済み浮動小数点 RGB テンソルを受け取ります。
高さと幅は 32 の倍数でなければならず、生成されるメモリには設定されたクエリ位置数以上が必要です。
モデルは入力のリサイズも正規化も行いません。

`ModelOutput.final` は最後に実行したデコーダ層を選択し、生のクラスロジット `[B,Q,K]` と
正規化済み cxcywh ボックス `[B,Q,4]` を含みます。完全な出力には勾配可能なエンコーダ予測と、
オプションの AIFI 特徴も含まれます。`DecoderPlan()` はデフォルトで 6 層すべてを実行・収集します。

<a id="python-api"></a>

## Python API

画像サポートをインストールした後、`image.jpg` と変換済み native state dict
checkpoint を用意します。任意のディレクトリで公式 checkpoint をダウンロードし、
インストール済みのコマンドで変換します。

```sh
curl --fail --location --create-dirs \
  --output official.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

checkout の開発環境では、インストール済みコマンドの代わりに
`uv run clean-rtdetr-convert` を使います。

converter は変換先パス、変換後ファイルの SHA-256、公式 load report を
含む JSON を出力します。既定ではベースラインに固定した EMA source と公式 SHA-256 を
検証します。別の対応 checkpoint を検証する場合は `--source` と
`--expected-sha256` で上書きできます。パッケージの loader では
`expected_sha256` キーワードを変換後ファイルの検証用に任意で指定できます。
画像推論 CLI は変換後の checkpoint を読み込み、ハッシュ指定のオプションは持たない。

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images

model = rtdetr_r50vd().eval()
load_model_weights(
    "native.pth",
    model=model,
)
with Image.open("image.jpg") as image:
    batch = preprocess_images([image])
with torch.inference_mode():
    output = model(batch.images, plan=DecoderPlan(collect_intermediate=False))
detections = postprocess(output.final, batch.original_sizes_wh)
```

`preprocess_images` は Pillow 画像および CPU RGB `uint8 [3,H,W]` テンソルの空でないシーケンスを受け付け、
サイズの異なる画像も扱えます。どちらの経路も Pillow の RGB 変換と 640×640 へのバイリニアリサイズを行い、
その後 CPU FP32 の `[0,1]` に変換します。レターボックス、平均・標準偏差による正規化、EXIF 回転はありません。

`PreprocessedImages.images` の形状は `[B,3,640,640]` です。`original_sizes_wh` は W,H 順の CPU int64
`[B,2]` です。`images` はモデルデバイスへ明示的に移動してください。後処理は必要に応じてサイズテンソルを移動します。

<a id="postprocessing"></a>

## 後処理

`postprocess(predictions, original_sizes_wh, top_k=300)` は、0 始まりの `labels[B,T]`、sigmoid 済み
`scores[B,T]`、元画像ピクセル座標の `boxes_xyxy[B,T,4]` を返します。すべてのクエリ・クラス組を順位付けするため、
1 クエリが複数のラベルを生成することがあります。同点の順序は PyTorch に従います。

上流互換経路では NMS、スコア閾値、ボックスのクリップを行いません。ラベルはモデルのクラスインデックスであり、
COCO カテゴリ ID ではありません。`top_k` は 1 以上、クエリ・クラス組数以下でなければなりません。

<a id="command-line-example"></a>

## コマンドライン例

`clean-rtdetr-predict` は 1 つ以上の画像パスとローカルの native state dict を受け取ります。
画像ごとの検出候補を JSON で標準出力に表示し、6 つのデコーダ層をすべて実行します。既定では全 300 候補を保持します。この例は画像や重みをダウンロードせず、
`--device` の既定値は `cpu` です。

```sh
clean-rtdetr-predict image.jpg \
  --checkpoint native.pth --conf-thresh 0.5 --visualize-dir outputs
```

`--conf-thresh` は指定したスコア以上の候補を残す。指定できる値は `[0, 1]` の有限値で、
既定値は `0`。top-k 選択後に絞り込み、スコア順を保持する。該当する候補がなければ、
その画像の `labels`、`scores`、`boxes_xyxy` は空配列となる。

`--visualize-dir` を指定すると、標準出力と同じ候補のボックス・COCO クラス名・スコアを
描画した PNG 画像を保存する。保存先ディレクトリは自動で作成し、元画像のサイズを保持する。
ファイル名は入力順の 1 始まりの番号と画像名を組み合わせた
`outputs/0001-image.png`、`outputs/0002-image.png` などとなる。
再実行時には既存の可視化ファイルを上書きするが、保存先がいずれかの入力画像と一致する場合は推論前にエラーとなる。
候補がなければ描画せずに画像を保存する。描画時の座標は画像内に収めるが、JSON の座標は
後処理の値をそのまま保持する。これらの CLI オプションは Python の `postprocess` API や COCO 評価経路には影響しない。

<a id="fixed-shape-onnx-export"></a>

## 固定形状 ONNX エクスポート

`clean_rtdetr_export` はテンソル専用ラッパー `FixedShapeRTDETR` と、
`export_onnx(model, destination, *, input_shape=(1, 3, 640, 640))` API を提供します。既定プロファイルは
バッチ 1、640×640、CPU FP32、ONNX opset 18 です。生のクラスロジット `[1,300,80]` と正規化 cxcywh ボックス `[1,300,4]` を出力し、
seed 41 と 42 について eager PyTorch と ONNX Runtime の一致を `atol=1e-5`、`rtol=1e-4` で検証します。
検証済みプロファイルではエクスポート時に `optimize=False`、ONNX Runtime のグラフ最適化無効を使用します。
既定の自動最適化は同じ入力で数値ゲートに失敗しました。このプロファイルは最適化実行を検証しません。

ラッパーは eval 状態の CPU FP32 コピーを保持し、元モデルの重みと学習状態を変更しません。`Float32` の jaxtyping 注釈は
入出力テンソルを記述し、明示的なチェックは不正な形状、dtype、学習モードも拒否します。組み込みの selector と decoder 型が必要なため、
forward メソッドを上書きしてもエクスポート動作を暗黙に変更できません。小さい固定形状は回帰テスト用に受け付けますが、記録済み R50vd
プロファイルは 640×640 です。

検証済み設定で成果物を実行するには、次を使います。

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "outputs/rtdetr.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)
# images: CPU float32 [1,3,640,640], RGB in [0,1].
logits, boxes = session.run(["logits", "boxes"], {"images": images.numpy()})
```

インストール済み CLI は `clean-rtdetr-export` で実行します。`--checkpoint` と `--output` は必須です。
`--expected-sha256` は native checkpoint の検証用に任意で指定できます。`--report` で JSON レポートのパスを指定できます（既定は output に `.json` を付けたパス）。
ローカルにある native checkpoint だけを読み込み、指定された場合は SHA-256 を検証します。重みはダウンロードしません。可変・動的形状、TensorRT、AP、
レイテンシは対象外です。

<a id="official-weights-and-parity"></a>

## 公式重みとパリティ

converter は core の内部 CLI 実装であり、公開 compatibility API はありません。公式互換ローダー `tools.compatibility.load_official_model_weights` と
`tools.compatibility.load_official_stage_weights` は checkout 専用です。元の
official envelope に `model` または `ema` の source と期待 SHA-256 を指定して使います。
パッケージの `clean_rtdetr.checkpoint.load_model_weights(path,
model=model, expected_sha256=...)` は converter が作成した plain native
state dict を受け取ります。

チェックポイント URL、ハッシュ、ソースは `benchmarks/baseline.json` に記録されています。実装済み推論ステージをすべて比較するには、次を実行します。

```sh
uv run python tools/checkpoint_parity.py \
  checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --full-model --report outputs/full-model-parity.json
```

先にレポートの親ディレクトリを作成してください。このコマンドは CPU FP32 の seed 付き合成 640×640 入力を比較し、データセット精度は測定しません。
`--full-model` を省略すると、合成特徴によるステージ比較になります。記録済み結果は `benchmarks/full-model-parity.json` にあります。
