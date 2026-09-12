# RT-DETRv3-inspired recipe

インストール済みの Clean RT-DETR に、RT-DETRv3 の学習分岐を注入する
独立した uv プロジェクトである。公開 stage API、`TrainingModel`、
`TrainingCriterion`、既存 Lightning adapter を使う。

この recipe は **RT-DETR + SAP + Transformer O2M + 簡略CNN/TAL補助ヘッド** である。
公式 RT-DETRv3 の完全再現ではない。公式 PP-YOLOE ヘッド、ATSS から
Task-Aligned assignment への切り替え、公式データ拡張・学習スケジュールは含まない。
公式 checkpoint 互換性、COCO AP、収束、推論速度、ONNX export は未検証である。
参照実装と相違点は [NOTICE.md](NOTICE.md) に記録している。

この recipe のソースは Apache-2.0 である。再配布時は LICENSE と NOTICE.md を保持する。
外部の重み・データ・依存パッケージにはそれぞれの条件が適用される。
詳細は[ライセンスの適用範囲](../../docs/ja/licensing.md)を参照する。

## 実行

Python 3.13 以降と uv を使う。

```sh
cd /path/to/clean-rtdetr/recipes/rtdetrv3
uv sync --locked
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic
```

core / training は `../../packages/core` と `../../packages/training` を
editable install する。workspace には参加せず、依存はこのディレクトリの
`pyproject.toml` と `uv.lock` で管理する。別の場所にコピーする場合は、
path source を checkout の絶対パスや対応する Git revision に変更して `uv lock` を実行する。

デフォルトは R18vd、person 1クラス、ランダム初期化、単一プロセス CPU FP32、
1 epoch である。synthetic は160角の2画像で、学習の接続確認に使う。

実画像にはライブラリの Penn-Fudan dataset を使う。

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1
```

`data.root` の直下に `PNGImages/*.png` と `PedMasks/*_mask.png` が必要である。
ダウンロードは行わない。入手方法は
[benchmark ドキュメント](../../packages/training/benchmarks/penn_fudan.md)を参照する。
既定の `seed=41`、`data.val_size=50` で train 120画像／val 50画像に分割し、
画像を640角にする。CLI は train のみ実行し、AP は計算しない。

## 学習分岐

- 通常 O2O: 既存 RT-DETR の selector と decoder。推論時はこの分岐だけを実行する。
- SAP: 独立した selector を3個追加し、それぞれ300 queryを選ぶ。
  matching query 間の attention edge を確率0.1で遮断し、同じmaskを全decoder層で使う。
  DN mask は維持し、各groupで独立した Hungarian matching を行う。
  DN 出力の分割・query 数の検証は training パッケージの公開
  `split_denoising_predictions` を使う。
- Transformer O2M: 独立 selector で450 queryを選び、GTを4回繰り返して
  Hungarian matching を行う。DN は追加しない。
- CNN補助: encoder の各特徴mapに畳み込みヘッドを接続する。
  Task-Aligned assignment、Varifocal、GIoU、DFL を使う。
  公式 PP-YOLOE ヘッドに代わる、このrecipe固有の小さな実装である。

backbone / encoder / projector は1回だけ実行する。decoder とその予測headは
すべてのquery groupで共有する。groupを別々にdecoderへ渡すことで、
group間のattentionを遮断する。SAP／O2Mのquery contentは、detachしたmemoryから
共有のLinear + LayerNormで作り、そのmapping自体には勾配が流れる。

通常O2OとSAPのlossはgroup数で平均し、O2MとCNNのlossを加算する。
各decoder層とencoder補助出力を教師あり学習し、O2Mは繰り返したGT数で正規化する。
返却lossは重み適用済みである。キー末尾は `_o2o_N`、`_o2m`、`_cnn` である。
各 query group は `TrainingOutput(predictions, denoising)` を返し、
matching 予測と DN の割り当てを training 側で保持する。
core の decoder は組み立てた全 query slot を返し、recipe が分割後の
`TrainingOutput` 全体を `SetCriterion` に渡す。

設定例:

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic module.training_model.sap_groups=1 \
  module.training_model.mask_probability=0.2 module.criterion.repeats=2
```

`sap_groups=0` でSAPを無効化できる。`num_denoising=0` はDNを無効化する。
`weight_o2o` / `weight_o2m` / `weight_cnn` は各lossの係数である。
画像を小さくする場合、projected memory の位置数が query 数以上になるようにする。

## 再開と検証

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic resume=outputs/rtdetrv3/checkpoints/last.ckpt trainer.max_epochs=2

uv run --locked pytest tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

再開時は初回と同じdata/model/loss設定を指定する。SAP確率、group数、
GT繰り返し数などの変更はcheckpoint設定照合で拒否される。
テストはattention mask、loss正規化、空GT、各分岐の勾配、推論分岐の分離、
Penn-Fudan設定注入を対象とする。
