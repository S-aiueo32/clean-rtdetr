# 実装ロードマップ

[English version](../roadmap.md)

設計提案はアーキテクチャ上の目標です。このファイルでは、小規模な stage のテストに
合格したことが完全な再現を意味しないよう、実装済みの範囲を分けて記録します。uv workspace
は core、固定形状のエクスポート、学習を別配布パッケージとして公開し、
`clean-rtdetr` への一方向の依存関係を持ちます。

## 現在の実装

PyTorch の推論経路は、images → R50vd backbone → HybridEncoder → 投影済み memory →
選択 query → 反復的な decoder prediction を接続しています。

| 領域 | 実装・検証済みの範囲 | 詳細 |
| --- | --- | --- |
| モデル stage | jaxtyping annotation と eager validation による固定 tensor contract、row-major memory projection と level offset、upstream mask を保つ anchor と top-k selection | [拡張ガイド](extensions.md) |
| Decoder と勾配 | 明示的な execution/output plan、self-attention、multi-scale deformable cross-attention、反復 refinement、content/reference detach 境界、微分可能な encoder auxiliary prediction | [勾配の所有権](extensions.md#gradient-ownership) |
| 構成 | component を注入できる `RTDETR`、公式 RT-DETR/RT-DETRv2 PResNet 10 種の名前付き architecture factory、optional AIFI feature、model と学習 component 用の階層的 Hydra config | [拡張ガイド](extensions.md#名前付きモデルプリセット) |
| Checkpoint と推論 | 公式推論 weight の strict loading、CPU FP32 full-model parity、Pillow/uint8 preprocessing、upstream 互換 top-k postprocessing | [Inference I/O](inference.md) |
| COCO 評価 | synthetic annotation でテストした prediction JSON と optional bbox metric adapter。pycocotools は metric の場合のみ読み込む | [COCO 評価](evaluation.md) |
| ONNX エクスポート | 固定 batch-1 640×640 向けの `FixedShapeRTDETR` と CPU FP32 ONNX Runtime parity。graph optimization は無効化 | [エクスポート設定](inference.md#fixed-shape-onnx-export) |
| Matching と loss | decoder/encoder auxiliary supervision と CPU loss/gradient parity を含む single-process Hungarian matching と Varifocal/L1/GIoU loss | [学習 loss](training.md) |
| Denoising | contrastive query、training wrapper に登録された class embedding、CPU upstream parity を持つ layer ごとの DN loss | [Contrastive denoising](training.md#contrastive-denoising) |
| 学習 loop | baseline AdamW group、EMA、CPU FP32 synthetic epoch-boundary resume を備えた Lightning adapter | [Lightning 学習](lightning.md) |

Forward と gradient parity のテストは SHA-256 で固定した upstream method を使います。
全データセットでの学習、GPU/DDP/AMP の検証、research method の構成、
全データセットでの COCO AP 測定、latency は未完了です。dynamic-shape ONNX と TensorRT は
検証済みエクスポートの範囲外です。

## 最初の段階

- [x] private repository、Python package、分離された開発環境。
- [x] 設計提案を保持し、upstream source の provenance を固定。
- [x] feature contract、memory projection、anchor、query selection。
- [x] 実パラメータ登録と勾配を伴う score/box head injection。
- [x] 変更していない upstream method と stage output および勾配を比較。
- [x] iterative refinement と明示的な layer selection を備えた query assembly と decoder。
- [x] 公式 checkpoint、その hash、実装済み stage における model/EMA の選択を取得。
- [ ] 固定 GPU 環境で full R50vd stage fixture、COCO AP、latency を取得。

## 次の段階

1. [x] 明示的な AIFI feature 境界を含む PResNet と HybridEncoder を追加。
2. [x] image-to-prediction R50vd を接続し、strict checkpoint loading を全推論 stage に拡張。
3. [x] 固定した公式 artifact に対して CPU FP32 complete-model checkpoint/stage parity を検証。
4. [x] Pillow/uint8 preprocessing、top-k postprocessing、画像推論の例を追加。
5. [ ] category-ID mapping を含む固定した評価環境で COCO AP を取得。
   prediction JSON と optional metric adapter は実装済みで、synthetic annotation で
   テスト済みです。全データセットでの精度測定は未完了です。
6. [x] 固定形状 ONNX エクスポートと CPU FP32 runtime parity を追加。
7. [x] DN query、criterion、CPU loss/gradient parity を追加。
8. [x] synthetic CPU FP32 Lightning 学習と epoch-boundary resume を追加。
9. [ ] 全データセットでの学習と GPU/DDP/AMP 実行を検証。
10. [ ] research method composition を実装。

## 検証の証拠

single-process FP32/FP64 matching criterion は Hungarian assignment と Varifocal/L1/GIoU
loss を実装し、decoder と encoder の auxiliary supervision を含みます。contrastive DN
construction、training wrapper に登録された embedding、layer ごとの DN loss も実装済みです。
CPU upstream query/loss/gradient parity はオフラインテストでカバーされています。詳細は
[学習 loss](training.md) を参照してください。optional な [Lightning adapter](lightning.md)
は optimizer、scheduler、EMA、application RNG state を含む CPU FP32 の学習 loop と
epoch-boundary resume を追加します。階層的 Hydra config は baseline model stage と
optimizer/scheduler factory を注入します。全データセットでの学習、GPU/DDP/AMP validation、
research method の構成は未完了です。

image-to-prediction API は raw class logits と normalized cxcywh box を返します。分離された
image I/O は Pillow image または CPU uint8 RGB tensor を受け取り、同じ 640×640 Pillow resize
path を使います。postprocessing は score、class index、clip されていない元画像座標の xyxy
box を返します。オフラインテストは image I/O contract と固定した upstream postprocessor parity
をカバーしますが、dataset accuracy は測定しません。

小規模なオフラインテストは component integration、勾配、checkpoint validation をカバーします。
`tools/checkpoint_parity.py --full-model` は seed 付き synthetic 640×640 input で公式 weight
を検査します。この比較はデータセット評価、GPU fixture の取得、latency 測定とは別です。
[記録済み parity 結果](../../benchmarks/full-model-parity.json) には source hash、checkpoint
provenance、input hash、環境、stage ごとの誤差が含まれます。[固定 shape export 結果](../../benchmarks/onnx-parity.json)
には、exporter と runtime の graph optimization を無効にした seed 41/42 の CPU FP32 ONNX
Runtime gate が記録されています。これはより広い deployment phase を完了させるものではなく、
AP、latency、dynamic shape、TensorRT の対応を確立するものでもありません。

<a id="research-acceptance"></a>

## 研究拡張の受け入れ

shared decoder weight を使う学習専用 branch、feature distillation と controller state、
distribution regression、recipe/loss composition を検証します。各拡張は contract test、
再現、エクスポート、backend verification を独立に進めます。これらの小規模な検査が通った後、
公式 DEIM + D-FINE と RT-DETRv3/v4 の integration を開始します。
