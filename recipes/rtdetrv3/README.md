# RT-DETRv3-inspired recipe

インストール済みの Clean RT-DETR に、RT-DETRv3 の学習分岐を注入する
独立した uv プロジェクトです。公開 stage API、`TrainingModel`、
`TrainingCriterion`、既存 Lightning adapter を使います。

この recipe は **RT-DETR + SAP + Transformer O2M + 簡略CNN/TAL補助ヘッド** です。
公式 RT-DETRv3 の完全再現ではありません。公式 PP-YOLOE ヘッド、ATSS から
Task-Aligned assignment への切り替え、公式データ拡張・学習スケジュールは含みません。
公式 checkpoint 互換性、COCO AP、収束、推論速度、ONNX export は未検証です。
参照実装と相違点は [NOTICE.md](NOTICE.md) に記録しています。

この recipe のソースは Apache-2.0 です。再配布時は LICENSE と NOTICE.md を保持してください。
外部の重み・データ・依存パッケージにはそれぞれの条件が適用されます。
詳細は[ライセンスの適用範囲](../../docs/ja/licensing.md)を参照してください。

## 実行

Python 3.13 以降と uv を使います。

```sh
cd /path/to/clean-rtdetr/recipes/rtdetrv3
uv sync --locked
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic
```

core / training は `../../packages/core` と `../../packages/training` を
editable install します。workspace には参加せず、依存はこのディレクトリの
`pyproject.toml` と `uv.lock` で管理します。別の場所にコピーする場合は、
path source を checkout の絶対パスや対応する Git revision に変更して `uv lock` を実行します。

デフォルトは R18vd、person 1クラス、ランダム初期化、単一プロセス CPU FP32、
1 epoch です。synthetic は160角の2画像で、学習の接続確認に使います。

実画像にはライブラリの Penn-Fudan dataset を使います。

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1
```

`data.root` の直下に `PNGImages/*.png` と `PedMasks/*_mask.png` が必要です。
ダウンロードは行いません。入手方法は
[benchmark ドキュメント](../../packages/training/benchmarks/penn_fudan.md)を参照してください。
既定の `seed=41`、`data.val_size=50` で train 120画像／val 50画像に分割し、
画像を640角にします。CLI は train のみ実行し、AP は計算しません。

## 学習分岐

- 通常 O2O: 既存 RT-DETR の selector と decoder。推論時はこの分岐だけを実行します。
- SAP: 独立した selector を3個追加し、それぞれ300 queryを選びます。
  matching query 間の attention edge を確率0.1で遮断し、同じmaskを全decoder層で使います。
  DN mask は維持し、各groupで独立した Hungarian matching を行います。
  DN 出力の分割・query 数の検証は training パッケージの公開
  `split_denoising_predictions` を使います。
- Transformer O2M: 独立 selector で450 queryを選び、GTを4回繰り返して
  Hungarian matching を行います。DN は追加しません。
- CNN補助: encoder の各特徴mapに畳み込みヘッドを接続します。
  Task-Aligned assignment、Varifocal、GIoU、DFL を使います。
  公式 PP-YOLOE ヘッドに代わる、このrecipe固有の小さな実装です。

backbone / encoder / projector は1回だけ実行します。decoder とその予測headは
すべてのquery groupで共有します。groupを別々にdecoderへ渡すことで、
group間のattentionを遮断します。SAP／O2Mのquery contentは、detachしたmemoryから
共有のLinear + LayerNormで作り、そのmapping自体には勾配が流れます。

通常O2OとSAPのlossはgroup数で平均し、O2MとCNNのlossを加算します。
各decoder層とencoder補助出力を教師あり学習し、O2Mは繰り返したGT数で正規化します。
返却lossは重み適用済みです。キー末尾は `_o2o_N`、`_o2m`、`_cnn` です。
各 query group は `TrainingOutput(predictions, denoising)` を返し、
matching 予測と DN の割り当てを training 側で保持します。
core の decoder は組み立てた全 query slot を返し、recipe が分割後の
`TrainingOutput` 全体を `SetCriterion` に渡します。

設定例:

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic module.training_model.sap_groups=1 \
  module.training_model.mask_probability=0.2 module.criterion.repeats=2
```

`sap_groups=0` でSAPを無効化できます。`num_denoising=0` はDNを無効化します。
`weight_o2o` / `weight_o2m` / `weight_cnn` は各lossの係数です。
画像を小さくする場合、projected memory の位置数が query 数以上になるようにしてください。

## 再開と検証

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  data=synthetic resume=outputs/rtdetrv3/checkpoints/last.ckpt trainer.max_epochs=2

uv run --locked pytest tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

再開時は初回と同じdata/model/loss設定を指定します。SAP確率、group数、
GT繰り返し数などの変更はcheckpoint設定照合で拒否されます。
テストはattention mask、loss正規化、空GT、各分岐の勾配、推論分岐の分離、
Penn-Fudan設定注入を対象とします。
