# ドキュメント

[English](../README.md) | 日本語

Clean RT-DETR の推論、予測の評価、学習、拡張のためのガイドである。
Python と PyTorch の知識を前提とする。外部ファイルなしで CPU のサンプルを動かすには、
[クイックスタート](../../README.ja.md#クイックスタート)から始める。

## 目的からガイドを選ぶ

| 目的 | ガイド |
| --- | --- |
| 重みを変換し、画像から予測する、または ONNX を出力する | [推論とエクスポート](inference.md) |
| COCO 予測 JSON を生成し、bbox 指標を計算する | [COCO 評価](evaluation.md) |
| ターゲットを準備し、マッチング・損失・denoising query を使う | [学習損失](training.md) |
| 学習を設定し、エポック境界から再開する | [Lightning による学習](lightning.md) |
| CUDA・AMP・DDP の実行制限と測定結果を確認する | [GPU 検証](gpu-validation.md) |
| モデルの部品やテンソルの契約を変更する | [拡張ガイド](extensions.md) |
| 公開 API で研究用 recipe を組み立てる | [recipe の責務分担](recipe-boundaries.md)と[実装例](../../recipes/README.md) |
| パッケージを開発・テスト・ビルドする | [開発への参加](../../CONTRIBUTING.ja.md) |
| ソース・重み・データセット・チェックポイントのライセンスを確認する | [ライセンスの適用範囲](licensing.md) |

## 環境を準備する

Python 3.13 または 3.14 と uv を使い、リポジトリのルートから実行する。
目的に応じてセットアップを選ぶ。

| 目的 | セットアップコマンド |
| --- | --- |
| 全 3 パッケージを開発・テストする | `uv sync --all-packages` |
| コアのテンソル API を使う | `uv sync --package clean-rtdetr` |
| 画像を読み込み、検出結果を生成する | `uv sync --package clean-rtdetr --extra inference` |
| 保存済み予測から COCO 指標を計算する | `uv sync --package clean-rtdetr --extra evaluation` |
| 画像推論と COCO 指標を併用する | `uv sync --package clean-rtdetr --extra inference --extra evaluation` |
| ONNX を出力し、CPU 実行結果の一致を確認する | `uv sync --package clean-rtdetr-export` |
| マッチング・損失・denoising・Lightning を使う | `uv sync --package clean-rtdetr-training` |

パッケージを指定した sync では、共有 `.venv` から他のメンバーや extra が削除される場合がある。
作業を切り替える際はセットアップを再実行し、ワークスペース全体の検証前には
`uv sync --all-packages` で環境を戻す。`evaluation` extra は任意であり、
`--all-packages` には含まれない。その後の `uv run` はインストール済みの環境を使う。
依存関係を変更した場合は sync を再実行する。recipe には個別の環境と手順がある。

画像推論とチェックポイントを使ったエクスポートには、ローカルの native state dict が必要である。
`clean-rtdetr-convert` による変換は[推論ガイド](inference.md)を参照する。
COCO 予測には画像とアノテーションも必要である。各コマンドはこれらをダウンロードしない。

## 対応状況を確認する

[ロードマップ](roadmap.md)は、実装済みの機能、検証結果、残る作業を分けて示す。
CPU での一致検証、過去の学習済み重みによる COCO 評価、範囲を限定した GPU 検証へのリンクから、
自分の用途に必要な対応状況を確認できる。

[設計提案](design-proposal.md)は、目標とするアーキテクチャと今後のデプロイ対応を説明する。
提案中の API や受け入れ基準は、実装済みの機能や測定結果ではない。
