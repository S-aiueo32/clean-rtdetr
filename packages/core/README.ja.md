# Clean RT-DETR Core

[English version](README.md)

`clean-rtdetr` は、公式 RT-DETR / RT-DETRv2 PResNet プリセットの読みやすい PyTorch constructor、テンソルコントラクト、画像前処理、top-k 後処理、native checkpoint の読み込み、COCO 予測/評価アダプターを提供します。Python 3.13 以降が必要です。

パッケージのインストールコマンドは次のとおりです。

```sh
python -m pip install clean-rtdetr
python -m pip install "clean-rtdetr[inference]"   # Pillow image I/O
python -m pip install "clean-rtdetr[evaluation]"  # pycocotools metrics
```

基本インストールだけで公式 R50vd checkpoint の変換を実行できます。Pillow、training、export の依存関係は不要です。

```sh
clean-rtdetr-convert --checkpoint official.pth --output native.pth
```

ソースから開発する場合は、リポジトリを clone してワークスペース環境をインストールします。

```sh
git clone https://github.com/S-aiueo32/clean-rtdetr.git
cd clean-rtdetr
uv sync --all-packages
uv run pytest packages/core/tests
```

core パッケージは、Pillow/uint8 前処理と COCO 形式の予測 JSON を含む CPU/PyTorch の画像から予測までの R50vd 推論に対応します。画像ファイルには `inference` extra を使います。パッケージの `clean_rtdetr.checkpoint.load_model_weights` は native plain `state_dict` を受け取り、SHA-256 は任意で検証できます。`clean-rtdetr-convert` は固定された公式 R50vd EMA source と hash を既定値にし、`--source` と `--expected-sha256` で上書きできます。converter は core の内部 CLI 実装であり、公開 compatibility API は提供しません。全体・stage・training の互換性ヘルパーは checkout 専用です。詳しくは [推論ガイド](../../docs/ja/inference.md) を参照してください。パッケージが重みをダウンロードすることはありません。固定形状 ONNX export は `clean-rtdetr-export` に、matching、loss、denoising、Lightning アダプターは `clean-rtdetr-training` に属します。データセット全体の学習、GPU/DDP/AMP 検証、COCO AP の測定、レイテンシベンチマークは検証済み範囲外です。

名前付きファクトリから、upstream の PResNet プリセット 10 種をランダム初期値で構築できます。`get_model_preset()` は対応する upstream YAML と checkpoint URL を来歴情報として返します。モデル構築時にダウンロードは行いません。

```python
from clean_rtdetr import create_model, list_models

print(list_models())
model = create_model("rtdetrv2_r18vd").eval()
```

列挙したプリセットではアーキテクチャ構築を検査しています。公式 checkpoint 変換、full-model parity、画像推論、ONNX parity の検証済み範囲は引き続き `rtdetr_r50vd` のみです。

チェックアウトの開発環境では、次のようにモデルを生成して前処理済みテンソルで実行できます（この例ではランダムな重みを使います）。

```python
import torch

from clean_rtdetr import DecoderPlan, rtdetr_r50vd

model = rtdetr_r50vd().eval()
with torch.inference_mode():
    output = model(torch.rand(1, 3, 640, 640), plan=DecoderPlan(collect_intermediate=False))
print(output.final.logits.shape, output.final.boxes.shape)  # [1, 300, 80], [1, 300, 4]
```

このパッケージは Apache-2.0 ライセンスで、固定した `lyuwenyu/RT-DETR` の revision を基にしています。ソースリポジトリの `third_party/RT-DETR` に upstream のソース、ライセンス、詳細な来歴、パリティテストを保持しています。
