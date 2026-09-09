# Clean RT-DETR Export

[English version](README.md)

`clean-rtdetr-export` は R50vd 検出器を固定形状 ONNX グラフへ export し、CPU FP32 の ONNX Runtime と PyTorch のパリティを確認します。Python 3.13 以降が必要で、対応する `clean-rtdetr` リリースに依存します。

```sh
python -m pip install clean-rtdetr-export
```

検証済み export プロファイルは、batch 1、RGB float32 入力 `[1, 3, 640, 640]`、CPU 実行、ONNX opset 18、logits `[1, 300, 80]` と正規化 cxcywh boxes `[1, 300, 4]` の raw 出力です。検証済みプロファイルでは exporter とランタイムのグラフ最適化を無効にします。export CLI にはローカルの公式チェックポイントと期待される SHA-256 が必要で、重みをダウンロードすることはありません。動的形状、TensorRT、AP、レイテンシは検証済み範囲外です。

ソースから開発する場合は次を実行します。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-export
uv run clean-rtdetr-export --help
uv run pytest packages/export/tests
```

チェックポイントを export し、検証レポートを書き出すには次を使います。

```sh
uv run clean-rtdetr-export \
  --checkpoint checkpoints/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --source ema \
  --expected-sha256 dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45 \
  --output outputs/rtdetr.onnx
```

Python API は `clean_rtdetr_export` から公開されています。`export_onnx` は構築済みの評価モデルを受け取り、`FixedShapeRTDETR` は lowering に使うテンソル専用ラッパーです。このパッケージは training パッケージをインストールせず、`clean_rtdetr.export` の互換性 shim も提供しません。

Clean RT-DETR は Apache-2.0 ライセンスで、コミット `068dfde65f2667ad6555883c69d73de886518cad` の `lyuwenyu/RT-DETR` を基にしています。ワークスペースのルートには upstream のソース、ライセンス、詳細な来歴が保持されています。
