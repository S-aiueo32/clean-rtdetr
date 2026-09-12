# ドキュメント

[English](../README.md) | 日本語

Clean RT-DETR は、コア検出器、ONNX エクスポート、学習を 3 つのパッケージに分けています。
まずは[リポジトリのクイックスタート](../../README.ja.md#クイックスタート)で、外部ファイルが不要な合成入力の CPU サンプルを実行できます。

## 環境を選ぶ

Python 3.13 または 3.14 と uv を使い、リポジトリのルートから実行してください。
目的に応じた環境を選びます。

| 目的 | セットアップコマンド |
| --- | --- |
| 全 3 パッケージの開発とテスト | `uv sync --all-packages` |
| コアのテンソル API を実行する | `uv sync --package clean-rtdetr` |
| 画像を読み込み、検出結果を生成する | `uv sync --package clean-rtdetr --extra inference` |
| 保存済み予測から COCO 指標を計算する | `uv sync --package clean-rtdetr --extra evaluation` |
| 画像推論と COCO 指標を併用する | `uv sync --package clean-rtdetr --extra inference --extra evaluation` |
| ONNX エクスポートと CPU 実行結果の一致を検証する | `uv sync --package clean-rtdetr-export` |
| マッチング・損失・ノイズ除去・Lightning を使う | `uv sync --package clean-rtdetr-training` |

パッケージを指定した sync では、共有 `.venv` から他のメンバーや extra が削除される場合があります。
別のガイドへ移る際には、その作業向けのセットアップを再実行してください。
ワークスペース全体の検証前には `uv sync --all-packages` で環境を戻します。
evaluation extra は任意であり、`--all-packages` には含まれません。
パッケージと extra は `uv sync` で選択し、その後のガイドのコマンドは `uv run` でインストール済みの環境を使います。
パッケージのメタデータや依存関係の選択を変更した場合は、`uv sync` を再実行してください。

## 利用ガイド

- [ライセンスの適用範囲](licensing.md): ソースのライセンス、帰属表示、外部データ・重み、teacher を含む checkpoint。
- [推論の入出力](inference.md): モデルと画像の契約、Python の推論例、チェックポイントの読み込み、後処理、固定形状 ONNX の実行。
- [COCO 評価](evaluation.md): ローカルデータセットの要件、予測 JSON、カテゴリ対応、部分集合の評価、指標レポート。
- [学習損失](training.md): ターゲット形式、Hungarian マッチング、重み付き損失、ノイズ除去クエリ、勾配と一致検証の保証範囲。
- [Lightning による学習](lightning.md): 合成データの学習、Hydra 設定、オプティマイザと EMA の状態、エポック境界での再開。

画像推論とチェックポイントを使ったエクスポートには、ローカルの変換済み native state dict が必要です。インストールした `clean-rtdetr` パッケージには、固定された公式 R50vd envelope 用の `clean-rtdetr-convert` コマンドが含まれます。全体・stage・training の互換性作業には checkout 専用の `tools.compatibility` ヘルパーを使います。公式 envelope の変換手順は [推論](inference.md) を参照してください。
COCO 予測には画像とアノテーションも必要です。各コマンドはこれらをダウンロードしません。
チェックポイントの来歴は[ベースラインのメタデータ](../../benchmarks/baseline.json)に記録しています。

## 開発と設計

- [開発への参加](../../CONTRIBUTING.ja.md): ワークスペース構成、ブランチ規則、必須検証、配布パッケージのビルド。
- [拡張ガイド](extensions.md): コンポーネントの注入、テンソル型注釈、勾配の責任範囲。
- [recipe の責務分担](recipe-boundaries.md): 4 つの recipe から抽出したライブラリ API と、ユーザーが所有するアルゴリズム・データ・状態・検証。
- [実装ロードマップ](roadmap.md): 完了済みの作業と残る検証。
- [設計提案](design-proposal.md): 目標とする研究用アーキテクチャと将来の統合。対応済み機能の一覧ではありません。

## 検証結果

[モデル全体のレポート](../../benchmarks/full-model-parity.json)は、シードを固定した合成入力で公式 R50vd 重みの出力を比較しています。
[ONNX レポート](../../benchmarks/onnx-parity.json)は、バッチ 1・640×640 固定形状で、グラフ最適化を無効にした CPU FP32 の一致検証を記録しています。
学習テストは CPU 上の損失、勾配、合成データによるエポック境界での再開を対象とします。
これらは収束やレイテンシを実証するものではありません。範囲を限定した GPU/DDP/AMP の検証は [GPU 実行](../gpu-validation.md)に記録しています。
別途、[COCO val2017 レポート](../../benchmarks/coco-val2017.json)に、公式学習済み R50vd EMA 重みを使った全 5,000 枚の CPU FP32 推論評価（AP 53.05）を記録しています。
[再現手順](../../benchmarks/coco-val2017.md)も参照してください。同一条件での上流実装との全データセット比較は未完了です。
[NVIDIA L4 native bf16 レポート](../../benchmarks/bf16-l4.md)と[機械可読な結果](../../benchmarks/bf16-l4.json)には、過去のソース snapshot に対するユーザー提供の単一 GPU 検証として、ランダム重みの R50 backbone と小規模 encoder/decoder による 64×64 推論、および raw denoising query 学習の smoke test が成功した結果を記録しています。これは COCO/AP 全体、性能、後続実装の Lightning GPU 対応を実証するものではありません。
