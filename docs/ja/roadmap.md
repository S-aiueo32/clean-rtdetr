# 実装ロードマップ

[English](../roadmap.md) | 日本語

実装済みの機能、測定結果、今後の作業を確認するためのページである。
より広いアーキテクチャの目標は[設計提案](design-proposal.md)を参照する。

## 現在の実装

推論経路は、画像 → backbone → HybridEncoder → projected memory → 選択した query →
decoder による反復予測を接続する。core・export・training は別の配布パッケージで、
export と training はそれぞれ core に依存する。

| 分野 | 実装範囲 | 詳細 |
| --- | --- | --- |
| モデル構成 | 部品を注入できる `RTDETR`、10 種類の RT-DETR/RT-DETRv2 PResNet プリセット、decoder の実行・出力計画、任意の AIFI 特徴出力 | [拡張](extensions.md) |
| 推論 | 公式 R50vd 推論重みの変換、native state dict の読み込み、Pillow/uint8 前処理、top-k 後処理 | [推論](inference.md) |
| 評価 | COCO 予測 JSON、category ID の変換、任意の bbox 指標計算 | [COCO 評価](evaluation.md) |
| エクスポート | `FixedShapeRTDETR`、バッチ 1・640×640 固定形状の CPU FP32 ONNX Runtime 実行、グラフ最適化は無効 | [エクスポート仕様](inference.md#fixed-shape-onnx-export) |
| 学習損失 | Hungarian マッチング、Varifocal/L1/GIoU、decoder/encoder の補助損失、native DDP での全 rank のターゲット数による正規化 | [学習損失](training.md) |
| Denoising | contrastive query、登録済み class embedding、query の分離、層ごとの DN 損失 | [Denoising](training.md#contrastive-denoising) |
| 学習ループ | Lightning、AdamW のパラメータグループ、EMA、CPU FP32/`bf16-mixed` と CUDA FP32/AMP、エポック境界での再開 | [Lightning](lightning.md)と[GPU の範囲](gpu-validation.md) |
| 研究用の例 | DEIM MAL、D-FINE FDR、RT-DETRv3 の学習分岐、RT-DETRv4 DSI/GAM の構成 | [Recipes](../../recipes/README.md) |

## 検証結果

各レポートは、記録されたソースのリビジョン・入力・環境に対する結果である。
合成入力でのテストに合格しても、全データセットの学習や精度を実証したことにはならない。

| 根拠 | 確認できること |
| --- | --- |
| [CPU モデル全体の一致検証](../../benchmarks/full-model-parity.json) | 公式 R50vd 重みとシード付き 640×640 合成入力での比較。重み・ソースのハッシュと段階ごとの誤差を記録 |
| [CPU ONNX の一致検証](../../benchmarks/onnx-parity.json) | exporter と runtime のグラフ最適化を無効にし、シード 41・42 で実行したバッチ 1・640×640 固定形状の FP32 比較 |
| [学習テスト](training.md#validation-scope)と[再開テスト](lightning.md#validation) | 固定した上流コードとの CPU 割り当て・損失・勾配・DN の一致検証、合成データでのチェックポイントと再開の検証 |
| [COCO val2017 評価](../../benchmarks/coco-val2017.md) | 過去の公式学習済み R50vd EMA 重みによる推論。全 5,000 枚の CPU FP32 bbox AP は 53.05。学習済み重みの評価記録 |
| [GPU 検証](gpu-validation.md) | 範囲を限定した CUDA 精度・学習・再開・native DDP の検証と制限 |
| [過去の L4 bf16 検証](../../benchmarks/bf16-l4.md) | ランダム重みの R50 backbone と小型 encoder/decoder による 64×64 推論、および当時の raw DN 学習。後続の Lightning GPU 対応の検証には適用できない |

## 残る作業

- 固定した GPU 環境で、R50vd 全体の段階別 fixture・COCO AP・レイテンシを取得する。
- 同一条件で、上流実装と COCO 全データセットの結果を比較する。
- 全データセットでの学習、収束、学習後の COCO AP を検証する。
- 公式研究手法、チェックポイント、学習スケジュールを再現する。
- 動的形状、分割エクスポート、TensorRT など、固定 CPU ONNX 仕様を超えるデプロイを検証する。

短い GPU/DDP/AMP 実行だけでは、学習全体の検証は完了しない。
過去の学習済み重みによる COCO 評価も、独立した学習再現を示すものではない。

<a id="research-acceptance"></a>

## 研究拡張の受け入れ

各拡張は、契約テスト、再現、エクスポート検証、backend 検証を独立して進める。
現在の recipe は構成例と限定的な数値・統合検証を示すもので、
公式 DEIM + D-FINE や RT-DETRv3/v4 の再現を実証するものではない。

新たな対応を主張する前に、設定・シード・ソースのリビジョン・成果物・比較結果を記録する。
拡張が所有する契約・損失・データ変換・状態は、[recipe の責務分担](recipe-boundaries.md)を参照して判断できる。
