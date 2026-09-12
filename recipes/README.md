# Recipes

recipe のソースは Apache-2.0 で提供する。外部の重み・データ・依存物と、
それらを含む学習成果物の扱いは[ライセンスの適用範囲](../docs/ja/licensing.md)を参照する。

recipe 実装から整理した[ライブラリとユーザーの責務分担](../docs/ja/recipe-boundaries.md)を参照する。
共通化した API と、各 recipe に残す数式・構成・状態管理・検証の判断を記録している。

インストール済みの Clean RT-DETR に、手元の実装を注入する利用例である。
各ディレクトリは利用者側のコード・設定・テストを持ち、ライブラリには同梱しない。
それぞれ独立した uv プロジェクトである。各ディレクトリで `uv sync --locked` を実行し、
`uv run --locked` から利用する。依存は各 `pyproject.toml` と `uv.lock` で管理する。
core / training は、この checkout のパッケージを path source で参照する。
デフォルトの dataset は公開 Penn-Fudan 実装を使い、
データを複製せず `data.root` で展開先を指定する。

各 recipe はパッケージとは別の PR で追加する。利用可能な実装の実行方法・
検証範囲・出典は、このディレクトリ内の各サブディレクトリの README.md と
NOTICE.md を参照する。recipe 同士の依存はない。
