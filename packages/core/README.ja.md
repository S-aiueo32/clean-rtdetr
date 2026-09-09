# Clean RT-DETR Core

[English version](README.md)

`clean-rtdetr` は、読みやすい PyTorch R50vd 参照モデル、テンソルコントラクト、画像前処理、top-k 後処理、公式チェックポイントの読み込み、COCO 予測/評価アダプターを提供します。Python 3.13 以降が必要です。

パッケージのインストールコマンドは次のとおりです。

```sh
python -m pip install clean-rtdetr
python -m pip install "clean-rtdetr[inference]"   # Pillow image I/O
python -m pip install "clean-rtdetr[evaluation]"  # pycocotools metrics
```

ソースから開発する場合は、リポジトリを clone してワークスペース環境をインストールします。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --all-packages
uv run pytest packages/core/tests
```

core パッケージは、Pillow/uint8 前処理と COCO 形式の予測 JSON を含む CPU/PyTorch の画像から予測までの R50vd 推論に対応します。画像ファイルには `inference` extra を使います。公式重みはローカルで用意し、期待される SHA-256 で検証する必要があります。パッケージが重みをダウンロードすることはありません。固定形状 ONNX export は `clean-rtdetr-export` に、matching、loss、denoising、Lightning アダプターは `clean-rtdetr-training` に属します。データセット全体の学習、GPU/DDP/AMP 検証、COCO AP の測定、レイテンシベンチマークは検証済み範囲外です。

チェックアウトの開発環境では、次のようにモデルを生成して前処理済みテンソルで実行できます（この例ではランダムな重みを使います）。

```python
import torch

from clean_rtdetr import DecoderPlan, rtdetr_r50vd

model = rtdetr_r50vd().eval()
with torch.inference_mode():
    output = model(torch.rand(1, 3, 640, 640), plan=DecoderPlan(collect_intermediate=False))
print(output.final.logits.shape, output.final.boxes.shape)  # [1, 300, 80], [1, 300, 4]
```

このパッケージは Apache-2.0 ライセンスで、コミット `068dfde65f2667ad6555883c69d73de886518cad` の `lyuwenyu/RT-DETR` を基にしています。ソースリポジトリには upstream のソース、ライセンス、詳細な来歴、パリティテストが保持されています。
