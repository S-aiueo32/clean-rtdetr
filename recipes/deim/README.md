# DEIM MAL recipe

インストールした `clean-rtdetr-training` に自分の criterion を注入し、
既存の Penn-Fudan dataset で学習する例である。
パッケージの変更、private メソッドの override、`tools/` への依存はない。

この例は **RT-DETR + DEIM の Matchability-Aware Loss (MAL)** である。
Dense O2O の Mosaic/MixUp、公式スケジュール、D-FINE の分布回帰は含まない。
DEIM 全体の再現や収束・COCO AP を検証した recipe ではない。
既存 Lightning adapter に合わせ、単一プロセス CPU FP32 で動作する。
式と参照 commit は [NOTICE.md](NOTICE.md) に記録している。

この recipe のソースは Apache-2.0 で、出典と変更範囲は [NOTICE.md](NOTICE.md) に
記録している。外部の重み・データ・依存パッケージにはそれぞれの利用条件が適用される。
リポジトリ全体の方針は [ライセンス案内](../../docs/ja/licensing.md) を参照する。

## インストールと実行

Python 3.13 以降と uv を使う。recipe ごとに `pyproject.toml`、`uv.lock`、
`.venv` を持つ独立プロジェクトである。リポジトリの workspace には参加しない。

```sh
cd /path/to/clean-rtdetr/recipes/deim
uv sync --locked
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1
```

core / training は `pyproject.toml` の path source で、この checkout の
`../../packages/core` と `../../packages/training` を参照する。
uv が各パッケージを recipe の `.venv` に editable install する。
別ディレクトリへ recipe をコピーする場合は、
依存元を checkout の絶対パスや対応する Git revision に変更し、`uv lock` を実行する。

デフォルトは R18vd・person 1クラス・Penn-Fudan の train split である。
上のコマンドは実画像1バッチで動作確認する。`+trainer.limit_train_batches=1`
を外すと train split 全体を1 epoch 学習する。初期重みはランダムである。

## Penn-Fudan dataset

この checkout の公開
`clean_rtdetr_training.data.PennFudanDetectionDataset` を
設定から直接使う。データの読み込みや mask から box への変換は複製しない。
`clean-rtdetr-training[benchmark]` を uv の依存に含めている。

`data.root` は展開済み `PennFudanPed` ディレクトリである。その直下に
`PNGImages/*.png` と `PedMasks/*_mask.png` が必要である。データを自動ダウンロード
しない。入手・検証手順は[benchmark ドキュメント](../../packages/training/benchmarks/penn_fudan.md)を参照する。

画像は640角の RGB float32、ラベルは person=0、box は正規化 cxcywh である。
既定の `seed=41`、`data.val_size=50` で train 120画像／val 50画像に分割する。
この CLI は train のみ実行し、val の AP は計算しない。

データなしで接続確認する場合は synthetic を明示できる。

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic
```

## 差し替える部分

- `deim_recipe/loss.py`: 通常の `nn.Module` として `MatchabilityAwareLoss` の式を実装。
- `deim_recipe/criterion.py`: `MALConfig` と `TrainingCriterion` を実装する
  `MALCriterion` を定義。公開 `iter_matched_stages`、`gather_matched_pairs`、
  `matched_iou_targets`、`matched_box_losses` を組み合わせ、MAL と L1/GIoU を計算する。
- `configs/data/penn_fudan.yaml`: インストール済み Penn-Fudan dataset の factory と split 設定。
- `configs/train.yaml`: `module.criterion._target_` が注入箇所。
  モデルと DN は Clean RT-DETR、optimizer と DataLoader は PyTorch、
  Trainer は Lightning の構成要素を使う。

MAL の変更例は `module.criterion.gamma=2.0`、モデルの変更例は
`module.training_model.model.name=rtdetr_r50vd` である。
criterion は既に重みを掛けた `loss_mal` / `loss_bbox` / `loss_giou` と
各 `_aux_N` / `_dn_N` を返す。
`DenoisingTrainingModel` の `TrainingOutput` 全体を criterion に渡すことで、
matching と DN の両方を学習する。式と重み、checkpoint の損失設定は
この recipe が所有し、ライブラリの `SetCriterion` に分類損失を注入しない。
単一の `DetectionOutput` を評価する場合は、`MALCriterion.stage_losses` を使う。

同じ設定で epoch 境界から再開する場合:

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed \
  resume=outputs/deim-mal/checkpoints/last.ckpt trainer.max_epochs=2 +trainer.limit_train_batches=1
```

MAL の gamma や重み、matcher の設定を変えた再開は拒否される。
実画像での再開には初回と同じ data/model/trainer overrides も指定する。

## recipe の検証

```sh
uv run --locked pytest tests
```

`uv sync --locked` はテスト用の dev 依存もインストールする。数値・勾配、空ターゲット、補助/DN loss、
Penn-Fudan の設定注入と checkpoint 設定を検証する。
