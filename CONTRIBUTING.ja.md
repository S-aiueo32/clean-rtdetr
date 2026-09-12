# コントリビューション

[English version](CONTRIBUTING.md)

このガイドでは、ワークスペースのセットアップ、目的を絞った変更、引き渡し前の検証を
説明する。Python、PyTorch、`uv` の基本的な利用を前提とする。

<a id="branch-strategy"></a>

## ブランチ戦略

このリポジトリでは、軽量なトランクベース開発を採用する。`main` は検証済みの状態を保ち、短命なブランチから目的を絞ったプルリクエストを通じて変更をマージする。恒久的な `develop` ブランチは作らない。

ブランチには次の接頭辞を使う。

- `feat/<issue>-<short-name>`: 新しい動作
- `fix/<issue>-<short-name>`: バグ修正
- `refactor/<short-name>`: 動作を変えない構造変更
- `test/<short-name>`: テストのみの変更
- `docs/<short-name>`: ドキュメントのみの変更
- `experiment/<topic>`: 一時的な研究実験

最新の `main` から開始し、`main` に直接 push せず、プルリクエストの目的を一つに保つ。マージには squash merge を使い、マージ後にブランチを削除する。研究結果には設定、シード、ソースコミットを記録する。実験ブランチ名だけでは再現性のメタデータにならない。

リリースには `v0.1.0` や `v0.1.1` のようなバージョンタグを使う。リリースに独立した安定化や保守が必要な場合に限り、リリースブランチを追加する。

<a id="development-setup"></a>

## 開発環境のセットアップ

Python 3.13 と 3.14 をサポートし、ローカル開発の既定値は 3.14 である。まず、ワークスペース全体をインストールする。

```sh
uv sync --all-packages
```

ワークスペースには、独立した `clean-rtdetr`、`clean-rtdetr-export`、
`clean-rtdetr-training` の各ディストリビューションがある。export と training は
core に依存するが、互いには依存しない。各メンバーは
`packages/<member>/` 以下の `pyproject.toml`、README、ソース、テストを所有する。
ルートの `pyproject.toml` はワークスペースと共有ツールの設定である。

実行時依存関係はそれを使うメンバーに置く。依存関係を変更したら、そのメンバーの
`pyproject.toml` を更新し、`uv lock` を実行して `uv.lock` の更新もコミットする。
共有する実験専用ツールはルートの `dev` グループに置く。

core 配布物には private 実装の `clean-rtdetr-convert` CLI が含まれる。checkpoint
変換には Pillow、export、training は必要ない。`tools/compatibility` の全体・
stage・training 用ヘルパーは checkout 専用で、インストール済みパッケージの公開 API
ではない。

各チェックアウトで一度、リポジトリのフックをインストールする。

```sh
uv run pre-commit install
```

フックは Ruff の修正を適用し、ty を実行する。すべての追跡ファイルには
`uv run pre-commit run --all-files` を実行できる。完全な pytest スイートは
引き渡し前に実行する。

<a id="development-smoke-test"></a>

## 開発用の動作確認

合成特徴量とランダムな重みでモデルの実行経路を確認する。画像、データセット、
チェックポイントは不要である。

```sh
uv run python tools/query_selection.py
```

memory、選択 query、decoder の tensor shape が出力される。これは実行経路と shape の
確認であり、検出精度を示すものではない。

<a id="validation"></a>

## 検証

次のコマンドでテストスイート、リンター、型チェッカー、サンプルを実行する。

```sh
uv run pytest \
  --cov=clean_rtdetr --cov=clean_rtdetr_export --cov=clean_rtdetr_training \
  --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python tools/query_selection.py
```

テストコマンドはカバレッジレポートも出力する。新しい動作には目的を絞ったテストを
追加する。これらの検証はテスト対象のコントラクトとパリティを示すもので、
データセット精度や学習の収束を示すものではない。引き渡し前には、
[ディストリビューション成果物](#distribution-artifacts) のビルドとメタデータの dry run
も実行する。

開発中に目的を絞ったテストを実行する例である。

```sh
uv run pytest \
  packages/core/tests/test_stages.py -k selection
```

Hypothesis テストは、入力サイズを制限した低コストなテンソルコントラクトとジオメトリ不変条件に絞り、完全なスイートを CI に適した状態に保つ。`.pytest_cache/`、`.hypothesis/`、`.coverage`、ツールのキャッシュなどのローカル成果物はコミットしない。

<a id="implementation-workflow"></a>

## 実装ワークフロー

新しいモデルステージでは、次の手順に従う。

1. 明示的なテンソルコントラクトを `packages/core/src/clean_rtdetr/contracts.py` に定義または更新する。
2. core のステージを `packages/core/src/clean_rtdetr/models` に、export の動作を `packages/export/src/clean_rtdetr_export` に、training の動作を `packages/training/src/clean_rtdetr_training` に実装する。
3. 所有するメンバーの `tests` ディレクトリに、コントラクト、勾配、形状を確認する目的を絞ったテストを追加する。
4. RT-DETR から派生したステージには、変更していない upstream コードとのパリティテストを追加する。
5. サポート範囲が変わったら、ロードマップと README を更新する。
6. 変更を引き渡す前に、完全な検証コマンドを実行する。

upstream 由来のソースと来歴は `third_party/RT-DETR` に置く。固定した upstream ソースを更新する場合は、`benchmarks/baseline.json` と `packages/core/tests/oracle.py` のコミットと SHA-256 を更新し、ライセンスと帰属表示を保持して、すべてのパリティテストを再実行する。oracle 実装をテスト対象のパッケージへ書き換えない。

core と training の CI は Python 3.13 と 3.14、export・評価・静的チェック・配布ビルドの CI は Python 3.13 で実行する。パッケージ境界のジョブはメンバーを明示的に選択する。CI はコミット済みの `uv.lock` が最新であることを検証するため `--locked` を渡す。たとえば export ジョブは次を使う。

```sh
uv sync --locked --package clean-rtdetr-export
uv run pytest \
  packages/export/tests/test_export.py packages/export/tests/test_export_cli.py
```

CI は core、training、export、評価、静的チェック、配布、recipe ごとの workflow に分かれる。
PR と `main` への push では GitHub の native path filter で対象を選び、複数の変更パスでは
対象をすべて実行する。default branch に入った workflow は `workflow_dispatch` でも起動
できる。non-main ref の古い実行はキャンセルし、`main` の実行はすべて保持する。

変更パスと実行対象の対応は次のとおりである。複数のパスが変わった場合は、該当する対象をすべて実行する。

| 変更ファイル | 自動実行する検証 |
| --- | --- |
| core の実装 | core、training、export、評価、静的チェック、core 配布ビルド、全 recipe |
| training の実装 | training、静的チェック、training 配布ビルド、全 recipe |
| export の実装 | export、静的チェック、export 配布ビルド |
| パッケージのテスト | 所有するテストワークフローと静的チェック。core の COCO テストは評価を選択 |
| ルートの `configs/**` | training |
| `tools/compatibility/**` | 関連する core/training テストと静的チェック |
| `tools/query_selection.py` | core と静的チェック |
| `third_party/RT-DETR/**` | core と training のパリティ |
| `benchmarks/baseline.json` | core の checkpoint 来歴テスト |
| パッケージの README・ライセンス・データ | 該当パッケージの配布ビルド |
| Markdown 以外の recipe ファイル | 該当 recipe。Python・ツール設定の変更は静的チェックも実行 |
| ルートの `pyproject.toml`・`uv.lock`、パッケージの `pyproject.toml` | ワークスペースの全 8 ワークフロー。core/training のパッケージメタデータは全 recipe も実行 |
| ワークフローの YAML | 該当ワークフロー |
| ルート README、`docs/**` の Markdown、CONTRIBUTING、AGENTS、その他のベンチマーク報告 | なし |

locked 操作は共有ワークスペースを解決するため、ワークスペースのメタデータ変更では全 workflow を
選択する。recipe は独自の lock と tool 設定を持つため、無関係な package test、root lock、
export の変更では選択しない。core の変更は依存先を保守的に選択する。配布 job は test と
静的チェックから独立しており、core job は checkout 外に install した conversion command も確認する。

依存関係の追加やファイル移動時には、呼び出し側を含めて push と PR 両方の path filter を更新する。
rename は移動前と移動後の両方の path に影響する。全 workflow を実行したい場合や非常に大きな
差分では manual dispatch を使う。path filter で skip される workflow は無条件の必須 check
にはできない。[GitHub の必須チェックに関する説明](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)を参照する。

ローカルでの合格だけでは、完全な検出器の AP、レイテンシ、チェックポイント、export のサポートを主張できない。これらの主張には対応する実装とフィクスチャが必要である。

Ruff は `ALL` を選択し、formatter/docstring の競合、pytest assertion、upstream の private
stage への access、CLI output、jaxtyping shape string（`F722`）には文書化された例外を設ける。
`F821` は有効なままである。ty は package の `src` と `tests`、`tools` を抑制なしで検査する。
変更していない vendored upstream source は両方の対象外である。

<a id="distribution-artifacts"></a>

## ディストリビューション成果物

3 つのメンバーは独立した PyPI ディストリビューションである。各メンバーの sdist と wheel の両方をビルドし、ローカルでメタデータを検証する。

```sh
uv build --package clean-rtdetr --no-sources --out-dir dist/core
uv build --package clean-rtdetr-export --no-sources --out-dir dist/export
uv build --package clean-rtdetr-training --no-sources --out-dir dist/training
uv publish --dry-run dist/core/*
uv publish --dry-run dist/export/*
uv publish --dry-run dist/training/*
```

生成されたアーカイブを調べ、それぞれが自身の Python 名前空間、パッケージデータ、README、ライセンス、来歴ファイルだけを含むことを確認する。公開名前空間は `clean_rtdetr`、`clean_rtdetr_export`、`clean_rtdetr_training` である。`clean_rtdetr.export` や `clean_rtdetr.training` の互換性 shim は復元しない。リリースバージョンはメンバーの依存関係制約を満たす必要があり、export または training を公開する前に、対象インデックスで `clean-rtdetr` が利用できなければならない。

<a id="implementation-notes"></a>

## 実装上の注意

COCO 予測生成は pycocotools なしで動作しなければならない。import と外部オブジェクト型は private な評価アダプター内に置き、公開インターフェースではプロジェクト所有のレコードと通常のメトリクス辞書を使う。オプションのメトリクステストは次で実行する。

```sh
uv sync --package clean-rtdetr --extra evaluation
uv run pytest \
  packages/core/tests/test_coco.py packages/core/tests/test_coco_cli.py
```

この extra は、コミット済みの uv 設定のもとで wheel のみを使う。[評価](docs/evaluation.md) を参照する。

バリデーターは eager なチェックとホストから見える形状メタデータを使う。固定形状 export アダプターは capture 前にコピーしたモデルを検証し、その後、専用メタデータを使って内部の attention、selection、refinement の算術を再利用する。これらの共有関数を変更するときも eager な検証を維持する。対応形状と、最適化を無効にした CPU FP32 ランタイムに必要な設定については、[export プロファイル](docs/inference.md#fixed-shape-onnx-export) を参照する。

テンソル境界では、`Float[Tensor, "batch queries 4"]` や `Int64[Tensor, "batch queries"]` のようなアノテーションを使う。ty は Python の型をチェックし、pytest フックは beartype と jaxtyping を使って実行時にアノテーション済みテンソルの形状と dtype をチェックする。既存の明示的なバリデーターは、オブジェクト間の不変条件と関数アノテーション外の検証を引き続き担当する。JAX は必要ない。[テンソルアノテーションの規約](docs/extensions.md#tensor-annotation-conventions) を参照する。
