# Clean RT-DETR

[English](README.md) | 日本語

読みやすさを重視し、研究用の拡張を組み合わせられる RT-DETR の参照実装です。
モデルの派生形、学習手法、実験レシピを、デプロイ用の変換から分離する設計を採用しています。

実装済みの機能、検証結果、今後の予定は[実装ロードマップ](docs/ja/roadmap.md)を参照してください。

## クイックスタート

Python 3.13 または 3.14 と uv を使用します。リポジトリのルートで実行してください。

```sh
uv sync --all-packages
uv run python tools/query_selection.py
```

この小規模な CPU サンプルには、画像、データセット、チェックポイントは不要です。
ランダムな重みで特徴量の射影、クエリ選択、デコードを実行し、次の形状を出力します。

```text
memory: (2, 336, 32)
query content: (2, 20, 32)
encoder auxiliary boxes: (2, 20, 4)
decoder logits: (2, 20, 80)
decoder boxes: (2, 20, 4)
```

実画像の検出には、ローカルに用意した公式チェックポイントを使用して
[推論ガイド](docs/ja/inference.md)に従ってください。ランダムな重みは API の動作確認用です。

## ドキュメント

| 目的 | ガイド |
| --- | --- |
| 実装状況と今後の予定を確認する | [実装ロードマップ](docs/ja/roadmap.md) |
| 環境を選ぶ・次に読むガイドを探す | [ドキュメント一覧](docs/ja/README.md) |
| 画像推論・公式重みの読み込み | [推論の入出力](docs/ja/inference.md) |
| ONNX モデルのエクスポートと実行 | [固定形状 ONNX エクスポート](docs/ja/inference.md#fixed-shape-onnx-export) |
| COCO 予測の生成・保存済み JSON の評価 | [COCO 評価](docs/ja/evaluation.md) |
| ターゲット・マッチング・ノイズ除去損失を扱う | [学習損失](docs/ja/training.md) |
| 合成データでの学習・チェックポイントからの再開 | [Lightning による学習](docs/ja/lightning.md) |
| モデル段階の拡張・勾配の境界を理解する | [拡張ガイド](docs/ja/extensions.md) |
| 開発・テスト・パッケージのビルド | [開発への参加](CONTRIBUTING.ja.md) |

## パッケージ

uv ワークスペースから、依存関係が一方向の 3 パッケージを公開します。

| 配布名 | Python パッケージ | ワークスペースメンバー | 責務 |
| --- | --- | --- | --- |
| `clean-rtdetr` | `clean_rtdetr` | `packages/core` | コアモデル・推論・前処理・評価 |
| `clean-rtdetr-export` | `clean_rtdetr_export` | `packages/export` | 固定形状 ONNX エクスポート・実行結果の一致検証 |
| `clean-rtdetr-training` | `clean_rtdetr_training` | `packages/training` | マッチング・損失・ノイズ除去・Lightning 学習 |

export と training は core に依存しますが、互いには依存しません。
以前の `clean_rtdetr.export` と `clean_rtdetr.training` 名前空間を互換エイリアスとして提供していません。
各メンバーには独自のメタデータ、README、ソース、テストがあり、個別にビルド・公開できます。
ルートの `pyproject.toml` はワークスペースと共通開発ツールを管理します。
リポジトリのルート自体は Python 配布パッケージではありません。

各パッケージの説明: [Core](packages/core/README.ja.md)、[Export](packages/export/README.ja.md)、[Training](packages/training/README.ja.md)。

## 開発への参加

開発環境、検証コマンド、静的検査の方針、実装上の注意点は
[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)を参照してください。

ローカルで簡単に確認するには、開発環境をインストールして次を実行します。

```sh
uv sync --all-packages
uv run pytest
```

ビルド、メタデータのドライラン、リリース順序は[配布成果物](CONTRIBUTING.ja.md#distribution-artifacts)を参照してください。

## 研究の方向性

[設計提案](docs/ja/design-proposal.md)では、目標とする
`ModelVariant × TrainingMethod × ExperimentRecipe` アーキテクチャを説明しています。
コンポーネントの境界は[拡張ガイド](docs/ja/extensions.md)、研究成果の受け入れ基準は
[実装ロードマップ](docs/ja/roadmap.md#research-acceptance)を参照してください。

## ライセンス

Clean RT-DETR は [Apache License, Version 2.0](LICENSE) で提供されます。
同梱した RT-DETR ソースとその改変には、上流の Apache-2.0 ライセンスと帰属表示を保持しています。
詳細は [third_party/RT-DETR/README.md](third_party/RT-DETR/README.md) を参照してください。

## 出典と来歴

[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) のコミット
`068dfde65f2667ad6555883c69d73de886518cad` に基づいています。
上流由来のモジュールには変更の告知を付けています。一致検証に使用する
[参照実装のソースとライセンス](third_party/RT-DETR/README.md)も保持しています。
[ベースラインのメタデータ](benchmarks/baseline.json)には、公式チェックポイントの URL、SHA-256、EMA の取得元を記録しています。
検証レポートとその対象範囲は[実装ロードマップ](docs/ja/roadmap.md)にまとめています。
