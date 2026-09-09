# コントリビューション

[English version](CONTRIBUTING.md)

<a id="branch-strategy"></a>

## ブランチ戦略

このリポジトリでは、軽量なトランクベース開発を採用しています。`main` は検証済みの状態を保ち、短命なブランチから目的を絞ったプルリクエストを通じて変更をマージします。恒久的な `develop` ブランチは作りません。

ブランチには次の接頭辞を使います。

- `feat/<issue>-<short-name>`: 新しい動作
- `fix/<issue>-<short-name>`: バグ修正
- `refactor/<short-name>`: 動作を変えない構造変更
- `test/<short-name>`: テストのみの変更
- `docs/<short-name>`: ドキュメントのみの変更
- `experiment/<topic>`: 一時的な研究実験

最新の `main` から開始し、`main` に直接 push せず、プルリクエストの目的を一つに保ちます。マージには squash merge を使い、マージ後にブランチを削除します。研究結果には設定、シード、ソースコミットを記録してください。実験ブランチ名だけでは再現性のメタデータになりません。

リリースには `v0.1.0` や `v0.1.1` のようなバージョンタグを使います。リリースに独立した安定化や保守が必要な場合に限り、リリースブランチを追加します。

<a id="development-setup"></a>

## 開発環境のセットアップ

Python 3.13 と 3.14 をサポートし、ローカル開発の既定値は 3.14 です。uv ワークスペースには `clean-rtdetr`、`clean-rtdetr-export`、`clean-rtdetr-training` の各ディストリビューションが含まれます。Export と training は core に依存しますが、互いには依存しません。各ディストリビューションは `packages/<member>/pyproject.toml`、README、ソースツリー、テストツリーを所有します。ルートの `pyproject.toml` はワークスペースと共有ツールの設定であり、公開可能なパッケージメタデータではありません。実行時依存関係はそれを使うパッケージに置きます。beartype、pytest、pytest-cov、Hypothesis、Ruff、ty は開発ツールです。Pillow はオプションの `inference` extra から利用でき、開発用の画像テストにもインストールされます。core のモデル import と後処理は Pillow から独立しています。開発用ワークスペース全体をインストールします。

```sh
uv sync --all-packages
```

core の基本配布物には内部実装の `clean-rtdetr-convert` CLI が含まれ、checkpoint
変換だけなら Pillow、export、training の依存関係は必要ありません。インストール済みの公開 surface には compatibility API を提供せず、全体・stage・training の互換性ヘルパーは checkout 専用の `tools.compatibility` にあります。

各チェックアウトで一度、リポジトリのフックをインストールします。

```sh
uv run pre-commit install
```

フックは Ruff の修正を適用し、コミット前に ty を実行します。すべての追跡ファイルに対して手動で実行するには、`uv run pre-commit run --all-files` を使います。完全な pytest スイートはコミットフックより高コストなため、CI/引き渡し時のチェックとして実行します。

`uv sync` はリポジトリの `uv.lock` を使い、プロジェクトの依存関係が変わると更新することがあります。パッケージの依存関係を変更する場合は、所有するメンバーの `pyproject.toml` を編集し、`uv lock` を実行して、生成された `uv.lock` の変更も含めます。共有の実験専用ツールは、メンバー自身が開発時または実行時に必要としない限り、ルートの `dev` 依存関係グループに置きます。

<a id="development-smoke-test"></a>

## 開発用の動作確認

合成特徴量とランダムな重みを使って、特徴量の射影、クエリ選択、デコードの実行経路を確認する。画像、データセット、チェックポイントは不要だ。セットアップ済みの環境で次を実行する。

```sh
uv run python tools/query_selection.py
```

成功時は次の形状が出力される。これは実行経路とテンソル形状が正しいことを確認するものであり、検出精度を示すものではない。

```text
memory: (2, 336, 32)
query content: (2, 20, 32)
encoder auxiliary boxes: (2, 20, 4)
decoder logits: (2, 20, 80)
decoder boxes: (2, 20, 4)
```

<a id="validation"></a>

## 検証

次のコマンドでテストスイート、リンター、型チェッカー、サンプルを実行します。

```sh
uv run pytest \
  --cov=clean_rtdetr --cov=clean_rtdetr_export --cov=clean_rtdetr_training \
  --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python tools/query_selection.py
```

テストコマンドはカバレッジレポートも出力します。新しい動作には、目的を絞ったテストを追加してください。スイートはコントラクトとパリティを検証しますが、データセット精度や学習の収束を示すものではありません。引き渡し前には、[ディストリビューション成果物](#distribution-artifacts) にあるビルドとメタデータの dry run も実行します。

開発中に目的を絞ったテストを実行する例です。

```sh
uv run pytest \
  packages/core/tests/test_stages.py -k selection
```

Hypothesis テストは、入力サイズを制限した低コストなテンソルコントラクトとジオメトリ不変条件に絞り、完全なスイートを CI に適した状態に保ちます。`.pytest_cache/`、`.hypothesis/`、`.coverage`、ツールのキャッシュなどのローカル成果物はコミットしないでください。

<a id="implementation-workflow"></a>

## 実装ワークフロー

新しいモデルステージでは、次の手順に従います。

1. 明示的なテンソルコントラクトを `packages/core/src/clean_rtdetr/contracts.py` に定義または更新します。
2. core のステージを `packages/core/src/clean_rtdetr/models` に、export の動作を `packages/export/src/clean_rtdetr_export` に、training の動作を `packages/training/src/clean_rtdetr_training` に実装します。
3. 所有するメンバーの `tests` ディレクトリに、コントラクト、勾配、形状を確認する目的を絞ったテストを追加します。
4. RT-DETR から派生したステージには、変更していない upstream コードとのパリティテストを追加します。
5. サポート範囲が変わったら、ロードマップと README を更新します。
6. 変更を引き渡す前に、完全な検証コマンドを実行します。

upstream 由来のソースと来歴は `third_party/RT-DETR` に置きます。固定した upstream ソースを更新する場合は、`benchmarks/baseline.json` と `packages/core/tests/oracle.py` のコミットと SHA-256 を更新し、ライセンスと帰属表示を保持して、すべてのパリティテストを再実行します。oracle 実装をテスト対象のパッケージへ書き換えないでください。

CI は Python 3.13 と 3.14 で同じチェックを実行します。パッケージ境界のジョブはメンバーを明示的に選択します。CI はコミット済みの `uv.lock` が最新であることを検証するため `--locked` を渡します。たとえば export ジョブは次を使います。

```sh
uv sync --locked --package clean-rtdetr-export
uv run pytest \
  packages/export/tests/test_export.py packages/export/tests/test_export_cli.py
```

ローカルでの合格だけでは、完全な検出器の AP、レイテンシ、チェックポイント、export のサポートを主張できません。これらの主張には対応する実装とフィクスチャが必要です。

Ruff は `ALL` を選択し、formatter/docstring の競合、pytest のアサーションと upstream の private メソッドへのアクセス、サンプル出力について文書化された例外を設けています。ハッシュ検証済み oracle ローダーには、固定したメソッドを実行するためのローカルな `S102` 例外が一つあります。jaxtyping の形状文字列を持つモジュールでは、これらの文字列が Python の forward reference ではないため `F722` を除外します。`F821` は有効なままです。ty は、ルールの抑制なしに、パッケージ内の `src` と `tests` ツリー、および `tools` をチェックします。変更していない vendored upstream ソースは、両方のツールのチェック対象外です。

<a id="distribution-artifacts"></a>

## ディストリビューション成果物

3 つのメンバーは独立した PyPI ディストリビューションです。各メンバーの sdist と wheel の両方をビルドし、ローカルでメタデータを検証します。

```sh
uv build --package clean-rtdetr --no-sources --out-dir dist/core
uv build --package clean-rtdetr-export --no-sources --out-dir dist/export
uv build --package clean-rtdetr-training --no-sources --out-dir dist/training
uv publish --dry-run dist/core/*
uv publish --dry-run dist/export/*
uv publish --dry-run dist/training/*
```

生成されたアーカイブを調べ、それぞれが自身の Python 名前空間、パッケージデータ、README、ライセンス、来歴ファイルだけを含むことを確認します。公開名前空間は `clean_rtdetr`、`clean_rtdetr_export`、`clean_rtdetr_training` です。`clean_rtdetr.export` や `clean_rtdetr.training` の互換性 shim は復元しないでください。リリースバージョンはメンバーの依存関係制約を満たす必要があり、export または training を公開する前に、対象インデックスで `clean-rtdetr` が利用できなければなりません。

<a id="implementation-notes"></a>

## 実装上の注意

COCO 予測生成は pycocotools なしで動作しなければなりません。import と外部オブジェクト型は private な評価アダプター内に置き、公開インターフェースではプロジェクト所有のレコードと通常のメトリクス辞書を使います。オプションのメトリクステストは次で実行します。

```sh
uv sync --package clean-rtdetr --extra evaluation
uv run pytest \
  packages/core/tests/test_coco.py packages/core/tests/test_coco_cli.py
```

この extra は、コミット済みの uv 設定のもとで wheel のみを使います。[評価](docs/evaluation.md) を参照してください。

バリデーターは eager なチェックとホストから見える形状メタデータを使います。固定形状 export アダプターは capture 前にコピーしたモデルを検証し、その後、専用メタデータを使って内部の attention、selection、refinement の算術を再利用します。これらの共有関数を変更するときも eager な検証を維持してください。対応形状と、最適化を無効にした CPU FP32 ランタイムに必要な設定については、[export プロファイル](docs/inference.md#fixed-shape-onnx-export) を参照してください。

テンソル境界では、`Float[Tensor, "batch queries 4"]` や `Int64[Tensor, "batch queries"]` のようなアノテーションを使います。ty は Python の型をチェックし、pytest フックは beartype と jaxtyping を使って実行時にアノテーション済みテンソルの形状と dtype をチェックします。既存の明示的なバリデーターは、オブジェクト間の不変条件と関数アノテーション外の検証を引き続き担当します。JAX は必要ありません。[テンソルアノテーションの規約](docs/extensions.md#tensor-annotation-conventions) を参照してください。
