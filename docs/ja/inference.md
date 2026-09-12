# 推論の入出力

[English version](../inference.md)

このガイドでは、対応する公式 R50vd 推論チェックポイントの変換、画像の予測、テンソルの
解釈、検証済み ONNX プロファイルのエクスポートを扱う。例では CPU FP32 の R50vd を
使用する。CUDA と AMP の検証済み範囲は [GPU 実行と検証](gpu-validation.md) を参照する。
ツールが画像や重みを自動ダウンロードすることはない。

## インストールとチェックポイントの準備

Python 3.13 以降と PyTorch が必要である。画像ファイルには Pillow、ONNX には別パッケージを
インストールする。

```sh
python -m pip install "clean-rtdetr[inference]"
python -m pip install clean-rtdetr-export  # 任意: ONNX export
```

ソースのチェックアウトでは `uv sync --package clean-rtdetr --extra inference` を使う。
エクスポートには `uv sync --package clean-rtdetr-export` を使う。以下の Python とパッケージ CLI の
コマンドには `uv run` を付けるが、`curl` には付けない。

同梱の変換コマンドは固定された公式 R50vd チェックポイント形式を受け取り、native plain
`state_dict` を書き出す。元のチェックポイントは自分でダウンロードしてから変換する。

```sh
curl --fail --location --create-dirs \
  --output official.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

既定では固定された EMA 重みを使い、変換コマンドが固定された SHA-256 を検証する。
JSON 出力には native ファイルのハッシュと公式読み込みレポートが含まれる。異なる検証要件の
対応チェックポイントを変換する場合だけ、`--source` と `--expected-sha256` を使う。
native loader の `expected_sha256` は任意である。画像予測 CLI にハッシュのオプションはない。

<a id="command-line-example"></a>

## 画像を予測する

`image.jpg` とローカルの `native.pth` を用意して、次を実行する。

```sh
clean-rtdetr-predict image.jpg \
  --checkpoint native.pth --conf-thresh 0.5 --visualize-dir outputs
```

このコマンドは 1 個以上の画像パスを受け取り、6 個の decoder layer をすべて実行し、入力ごとに
1 個の JSON レコードを標準出力へ書き出す。`--device` の既定値は `cpu` である。まず
query/class の組を上位 300 件選び、`--conf-thresh` は有限の `[0, 1]` 値（既定値 `0`）以上の
スコアを残す。候補が残らない入力では配列が空になる。

`--visualize-dir` は `outputs/0001-image.png` のような PNG を書き出す。標準出力と同じ
候補に COCO の名前とスコアを表示し、元画像の寸法を保つ。ディレクトリは作成されるが、
入力画像と同じ保存先は拒否される。再実行では可視化ファイルを置き換える。描画時の枠は
画像の範囲に切り詰めるが、JSON には後処理後の座標をそのまま保持する。

<a id="python-api"></a>

## Python API を使う

`preprocess_images` には Pillow が必要で、Pillow 画像または CPU 上の `uint8` RGB テンソル
`[3,H,W]` からなる空でないシーケンスを受け取る。すべて Pillow で RGB に変換し、640×640 へ
バイリニアリサイズして、`[0,1]` の CPU FP32 値を返す。レターボックス、平均/標準偏差による
正規化、EXIF 回転は行わない。

```python
import torch
from PIL import Image

from clean_rtdetr import DecoderPlan, rtdetr_r50vd
from clean_rtdetr.checkpoint import load_model_weights
from clean_rtdetr.inference import postprocess, preprocess_images

model = rtdetr_r50vd().eval()
load_model_weights("native.pth", model=model)

with Image.open("image.jpg") as image:
    batch = preprocess_images([image])
with torch.inference_mode():
    output = model(batch.images, plan=DecoderPlan(collect_intermediate=False))
detections = postprocess(output.final, batch.original_sizes_wh)
```

モデルを呼ぶ前に `batch.images` をモデルのデバイスへ移動する。返り値
`PreprocessedImages.images` は `[B,3,640,640]`、`original_sizes_wh` は `(width, height)` 順の
CPU int64 `[B,2]` である。`postprocess` は必要に応じてサイズテンソルを予測テンソルのデバイスへ移す。

<a id="model-tensors"></a>

## モデルと後処理の契約

`RTDETR` は前処理済みの浮動小数点 RGB テンソル `[B,3,H,W]` を受け取る。高さと幅は
32 の倍数で、投影後のメモリには設定した query position 数以上が必要である。モデル側で
リサイズや正規化は行わない。

`ModelOutput.final` は最後に実行した decoder layer の未処理 class logits `[B,Q,K]` と、正規化済み
`cxcywh` box `[B,Q,4]` である。`DecoderPlan()` は既定で 6 layer をすべて実行して収集する。
最終出力だけが必要なら `DecoderPlan(collect_intermediate=False)` を使う。

<a id="postprocessing"></a>

`postprocess(predictions, original_sizes_wh, top_k=300)` は、ゼロ始まりの
`labels[B,T]`、sigmoid 後の `scores[B,T]`、元画像のピクセル座標による `boxes_xyxy[B,T,4]` を返す。
query/class の組をすべて順位付けするため、1 個の query から複数の label が出ることがある。
NMS、スコアしきい値、切り詰めは意図的に適用しない。label は COCO category ID ではなく
モデルの class index である。`top_k` は 1 から `Q*K` の整数にする。同順位の順序は PyTorch に従う。

<a id="fixed-shape-onnx-export"></a>

## 固定形状 ONNX を export する

`clean_rtdetr_export.export_onnx(model, destination, *,
input_shape=(1, 3, 640, 640))` はテンソル専用 `FixedShapeRTDETR` グラフを書き出す。
検証済みプロファイルは、batch 1、CPU FP32 RGB 入力 `[1,3,640,640]`、opset 18、未処理 logits
`[1,300,80]`、正規化 `cxcywh` box `[1,300,4]` である。CLI はローカル native checkpoint を
読み込む。`--checkpoint` と `--output` は必須、`--expected-sha256` は任意で、`--report` の
既定値は出力パスの `.json` 接尾辞である。

```sh
clean-rtdetr-export --checkpoint native.pth --output outputs/rtdetr.onnx
```

エクスポートはシード 41 と 42 の合成入力で eager PyTorch と ONNX Runtime を検証する
（`atol=1e-5`、`rtol=1e-4`）。これが検証済みの数値プロファイルなので、`optimize=False` と
無効化した ONNX Runtime のグラフ最適化を使う。まず上の Python API の例を実行し、前処理済みの
画像を持つ `batch.images` を用意してから、同じ設定でグラフを実行する。

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "outputs/rtdetr.onnx", sess_options=options, providers=["CPUExecutionProvider"]
)
# batch.images は [0,1] の CPU float32 RGB テンソル [1,3,640,640]。
logits, boxes = session.run(["logits", "boxes"], {"images": batch.images.numpy()})
```

ラッパーは source model を CPU FP32 の評価モードにコピーするため、元の model は変わらない。
テンソルの形状、dtype、評価モードを検証する。より小さい固定形状は回帰テスト向けに使える。
記録済み R50vd プロファイルは 640×640 である。動的形状、TensorRT、AP、レイテンシは
この検証範囲外である。

<a id="official-weights-and-parity"></a>

## 公式重みとの互換性

`clean-rtdetr-convert` は公開コマンドラインインターフェースである。ただし、公式重みを変換する
実装は private であり、公開 compatibility API ではない。チェックアウトでは
`tools.compatibility.load_official_model_weights` と
`tools.compatibility.load_official_stage_weights` が、明示した `model` または `ema` source で
元の公式チェックポイント形式を読み込む。パッケージコードは
`clean_rtdetr.checkpoint.load_model_weights` で native ファイルを読み込む。

固定 URL、ハッシュ、source は
[`benchmarks/baseline.json`](../../benchmarks/baseline.json) にある。合成 CPU FP32 の parity 比較には
`tools/checkpoint_parity.py` を使う。`--full-model` はシード固定の 640×640 入力を比較するだけで、
データセットの精度は測定しない。full-model の記録結果は
[`benchmarks/full-model-parity.json`](../../benchmarks/full-model-parity.json) である。
