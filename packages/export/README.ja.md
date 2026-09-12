# Clean RT-DETR Export

[English version](README.md)

`clean-rtdetr-export` は R50vd 検出器を固定形状 ONNX グラフへ export し、CPU FP32 の ONNX Runtime と PyTorch のパリティを確認する。Python 3.13 以降が必要で、対応する `clean-rtdetr` リリースに依存する。

```sh
python -m pip install clean-rtdetr-export
```

検証済み export プロファイルは、batch 1、RGB float32 入力 `[1, 3, 640, 640]`、CPU 実行、ONNX opset 18、logits `[1, 300, 80]` と正規化 cxcywh boxes `[1, 300, 4]` の raw 出力である。検証済みプロファイルでは exporter とランタイムのグラフ最適化を無効にする。export CLI にはローカルの変換済み native state_dict が必要で、期待する SHA-256 は任意で検証できる。重みをダウンロードすることはない。公式 envelope は `clean-rtdetr-convert` で変換する（[推論ガイド](../../docs/ja/inference.md)）。動的形状、TensorRT、AP、レイテンシは検証済み範囲外である。

ソースから開発する場合は次を実行する。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --package clean-rtdetr-export
uv run clean-rtdetr-export --help
uv run pytest packages/export/tests
```

チェックポイントを export し、検証レポートを書き出すには次を使う。

```sh
uv run clean-rtdetr-export \
  --checkpoint checkpoints/rtdetr_r50vd.pth \
  --output outputs/rtdetr.onnx
```

Python API は `clean_rtdetr_export` から公開されている。`export_onnx` は構築済みの評価モデルを受け取り、`FixedShapeRTDETR` は lowering に使うテンソル専用ラッパーである。このパッケージは training パッケージをインストールせず、`clean_rtdetr.export` の互換性 shim も提供しない。

Clean RT-DETR は Apache-2.0 ライセンスで、コミット `068dfde65f2667ad6555883c69d73de886518cad` の `lyuwenyu/RT-DETR` を基にしている。ワークスペースのルートには upstream のソース、ライセンス、詳細な来歴が保持されている。
