# Recipes

recipe 実装から整理した[ライブラリとユーザーの責務分担](../docs/ja/recipe-boundaries.md)を参照してください。
共通化した API と、各 recipe に残す数式・構成・状態管理・検証の判断を記録しています。

インストール済みの Clean RT-DETR に、手元の実装を注入する利用例です。
各ディレクトリは利用者側のコード・設定・テストを持ち、ライブラリには同梱しません。
それぞれ独立した uv プロジェクトです。各ディレクトリで `uv sync --locked` を実行し、
`uv run --locked` から利用します。依存は各 `pyproject.toml` と `uv.lock` で管理します。
core / training は、この checkout のパッケージを path source で参照します。
デフォルトの dataset は公開 Penn-Fudan 実装を使い、
データを複製せず `data.root` で展開先を指定します。

各 recipe はパッケージとは別の PR で追加します。利用可能な実装の実行方法・
検証範囲・出典は、このディレクトリ内の各サブディレクトリの README.md と
NOTICE.md を参照してください。recipe 同士の依存はありません。
