# D-FINE FDR recipe

パッケージ内部を変更せず、手元の decoder / matcher / criterion を注入する例である。
既存の RT-DETR backbone・encoder・query stages と Lightning 学習ループに、
D-FINE の分布回帰を組み合わせる。

実装範囲は FDR の非一様 bin・残差分布更新・LQE、FGL、GO-LSD の
union matching と最終層からの DDF 自己蒸留である。VFL と L1/GIoU も含む。
公式 D-FINE の HGNetv2・attention gate・広い追加層・DN・augmentation・
学習スケジュールは含まない。公式モデルや重みとの互換性、COCO AP は未検証である。
検証範囲は単一プロセス CPU FP32 と、一部の FP64 数値チェックである。

## uv で実行

Python 3.13 以降と uv を使い、リポジトリ内で実行する。
`__init__.py` やリポジトリのソースを import path に追加する処理はない。

```sh
cd /path/to/clean-rtdetr/recipes/dfine
uv sync --locked
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1
```

独立した `pyproject.toml` / `uv.lock` / `.venv` を持ち、親 workspace には参加しない。
core / training は `../../packages/core` / `../../packages/training` の path 依存である。
uv がローカルのパッケージを editable でインストールするため、ソース変更も反映される。
recipe を別の場所へコピーする場合は path を調整する。

デフォルトはランダム初期化した R18vd ベース・person 1クラス・Penn-Fudan である。
上のコマンドは実画像1バッチの確認である。`+trainer.limit_train_batches=1` を外すと
train split 全体を1 epoch 学習する。収束や AP を検証した設定ではない。

## Penn-Fudan dataset

この checkout の公開 `clean_rtdetr_training.data.PennFudanDetectionDataset` を再利用する。
`clean-rtdetr-training[benchmark]` を uv の依存に含めている。
`data.root` は `PNGImages/` と `PedMasks/` を持つ展開済み `PennFudanPed` である。
入手・検証手順は[benchmark ドキュメント](../../packages/training/benchmarks/penn_fudan.md)を参照する。
データのダウンロードは自動では行わない。

読み込み・分割・640角への resize・mask から box への変換は、すべて既存 dataset が
担当する。dataset は model object なしで構築し、固定の person=0 を使う。
synthetic も公開 `SyntheticDetectionDataset` を使い、画像サイズ160とクラス数を
設定から明示的に渡す。

person=0、正規化 cxcywh、既定の `seed=41` / `data.val_size=50` で
train 120画像／val 50画像である。この CLI は train のみ実行し、val AP は計算しない。

データなしで接続確認する場合:

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic
```

モデルは `module.training_model.name=rtdetr_r50vd`、分布 bin は
`distribution.reg_max=16` のように変更できる。分布設定は decoder と criterion に
同じ値を渡す。片側だけ変えると学習開始時に拒否される。

同じ設定から epoch 境界で再開する場合:

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed \
  resume=outputs/dfine/checkpoints/last.ckpt trainer.max_epochs=2 +trainer.limit_train_batches=1
```

実画像での再開には初回と同じ data/model/trainer overrides も指定する。
モデル・criterion 設定の変更は checkpoint の互換性チェックで検出する。

## 注入の構成

- `model.py`: 既存の `create_model` が作る stages に `FDRDecoder` を渡して構成。
- `decoder.py`: 分布・固定 reference・初期予測を明示的に返す `DistributionOutput`。
  モジュールの隠れたキャッシュや forward hook で loss に渡す方式は使わない。
- `criterion.py`: 公開 `TrainingCriterion` を実装し、分布を使って loss を計算。
- `matcher.py`: `DFINEMatcher` が公開 `HungarianMatcher` の
  `prediction_box_range="unbounded"` を選び、FDR の `[0, 1]` を超える box を扱う。
  cost と solver はライブラリを再利用し、target の正規化検証は維持する。

予測の box は正規化座標系の cxcywh であるが、画像の外へ出ることがある。
この recipe の分布出力を既存の推論／export CLI に直接渡す対応は含まない。

## 検証と出典

```sh
uv run --locked pytest tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

非一様 bin・座標変換、VFL/L1/GIoU/FGL/DDF の値と分布勾配を、固定した
公式ソースの関数と比較する。モデル全体の upstream parity を示すものではない。
空ターゲット、raw box、union の割り当て、初期 box head の勾配、LQE の on/off、
decoder depth と設定の不整合も検証する。参照 commit と差分は [NOTICE.md](NOTICE.md)。

この recipe のソースは Apache-2.0 で、出典と変更範囲は [NOTICE.md](NOTICE.md) に
記録している。外部の重み・データ・依存パッケージにはそれぞれの利用条件が適用される。
リポジトリ全体の方針は [ライセンス案内](../../docs/ja/licensing.md) を参照する。
