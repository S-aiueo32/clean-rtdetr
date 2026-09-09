# RT-DETRv4 DSI + GAM recipe

公開 `after_aifi` 出力に RT-DETRv4 の Deep Semantic Injector (DSI) と
Gradient-guided Adaptive Modulation (GAM) を接続する学習例です。
core / training の変更、private API、特徴取得用の forward hook は必要ありません。

構成は **RT-DETRv2-R18vd + DN + VFL/L1/GIoU + DSI/GAM** です。
公式 RT-DETRv4-S/M/L/X の HGNetv2、D-FINE の分布回帰、DEIM の MAL・
augmentation・公式スケジュールは含みません。公式モデル全体の再現、
検出器重みの互換性、収束、COCO AP、推論速度は未検証です。
参照 commit と差分は [NOTICE.md](NOTICE.md) に記録しています。

この recipe のソースは Apache-2.0 です。再配布時は LICENSE と NOTICE.md を保持してください。
外部の重み・データ・依存パッケージにはそれぞれの条件が適用されます。
詳細は[ライセンスの適用範囲](../../docs/ja/licensing.md)を参照してください。

## インストールと接続確認

Python 3.13 以降と uv を使います。親 workspace に参加しない独立プロジェクトです。

```sh
cd /path/to/clean-rtdetr/recipes/rtdetrv4
uv sync --locked
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  teacher=smoke data=synthetic
```

core / training は `../../packages/core` / `../../packages/training` を
editable install します。recipe を別の場所へコピーする場合は
`pyproject.toml` の path source を変更して `uv lock` を実行してください。

`teacher=smoke` は凍結したランダムな小型畳み込みです。VFM の意味表現を持たず、
学習品質の評価には使いません。実画像での接続確認には `data=synthetic` を外して
`data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1` を指定します。

## 学習済み DINOv3 teacher

デフォルトはローカルの DINOv3 ViT-B/16 と Penn-Fudan です。
[DINOv3 の公式手順](https://github.com/facebookresearch/dinov3#pretrained-models)に従い、
公式 checkout と LVD-1689M の ViT-B/16 重みを用意してください。
teacher のコード・重みは同梱せず、自動ダウンロードも行いません。
DINOv3 のコードと重みには、recipe の Apache-2.0 とは別の
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md)
が適用されます。実際に取得する checkout と重みに付属する条件を確認してください。
`repo_dir` は `hubconf.py` があるディレクトリ、`weights` はローカルの `.pth` です。
実験時は利用した DINOv3 checkout の commit も記録してください。

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  teacher.repo_dir=/absolute/path/dinov3 \
  teacher.weights=/absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  data.root=/absolute/path/PennFudanPed +trainer.limit_train_batches=1
```

単一プロセス CPU FP32、person 1クラス、ランダム初期化した student です。
上のコマンドは実画像1バッチを学習し、batch 制限を外すと1 epoch 実行します。
Penn-Fudan の読み込み・mask から box への変換・分割は既存 dataset が担当します。
入手方法は[benchmark 手順](../../packages/training/benchmarks/penn_fudan.md)を参照してください。
画像は640角の RGB float32、person=0、box は正規化 cxcywh です。
既定の `seed=41` / `val_size=50` は train 120画像、val 50画像です。
この CLI は train のみ実行し、val AP は計算しません。

学習済み DINOv3 重みを使った実行は未検証です。teacher adapter は、ローカル Hub の
代替モデルで前処理・token layout・凍結・入力検証を確認しています。

## 注入と再開

- `model.py`: AIFI 出力を単一 Linear で teacher の channel 数へ投影し、
  `DistillationOutput` で student / teacher 特徴を criterion に渡します。
  DN wrapper の `TrainingOutput.predictions.after_aifi` から特徴を取得し、
  `TrainingOutput` 全体を検出 criterion に渡して DN supervision を保持します。
- `teacher.py`: ImageNet 正規化、2倍の平均 pooling、凍結 DINOv3 の patch token。
  画像の縦横は32の倍数が必要です。teacher は常に eval / 勾配なしです。
- `criterion.py`: teacher の bilinear resize と正規化 token の `mean(1-cosine)`。
  検出側の最終・補助・DN loss は既存 `SetCriterion` が担当します。
- `gam.py`: clipping 前の AIFI パラメータ勾配の L1 比率を epoch 内で平均し、
  公式コードの odds-ratio 更新で次 epoch の蒸留重みを調整します。
  `rho` / `delta` は百分率です。初期値は重み15、rho=2、delta=0.1です。
  パラメータの取得には公開 `HybridEncoder.aifi_parameters()` を使います。
- `configs/train.yaml`: 公開 adapter の `required_callback_keys` に
  `GradientAdaptiveModulation` を宣言し、再開時の callback 省略を検出します。

モデルは `module.training_model.student.model.name=rtdetrv2_r50vd`、
初期重みは `module.criterion.weight=10`、GAM の目標は
`trainer.callbacks.gam.rho=3.0` で変更できます。
`trainer.callbacks.gam.default_weight` は勾配比0の場合の復帰値です。
任意の `+trainer.callbacks.gam.stop_epoch=N` で、0始まり epoch N の終わりから
固定値に戻せます。既定では augmentation の停止スケジュールを持たず、GAM を継続します。

```sh
uv run --locked python -m clean_rtdetr_training.cli --config-dir configs --config-name train \
  teacher=smoke data=synthetic \
  resume=outputs/rtdetrv4/checkpoints/last.ckpt trainer.max_epochs=2
```

再開には初回と同じ teacher/data/model/trainer overrides を指定してください。
student・projector・teacher・蒸留重みは state dict、GAM の設定と観測値は
callback state に保存します。teacher のローカル重みの SHA256 も照合します。
GAM 設定変更、callback の省略、teacher／criterion 設定変更は再開時に拒否します。

DINOv3 を使う場合、再開用 checkpoint と学習モデル全体の state dict には
DINOv3 の重みも含まれます。EMA を有効にした場合はその状態にも含まれます。
これらを共有するときは DINOv3 License の再配布条件に従い、契約書も添付してください。
student detector だけを抽出すれば teacher tensor の同梱は避けられますが、
それだけで蒸留した重みのライセンスが確定するわけではありません。

`training_model.predict(images)` は teacher と projector を実行せず、
呼び出し前の train/eval 状態を保ちます。recipe の checkpoint や
`DistillationOutput` を既存推論／export CLI に直接渡す対応は含みません。

## 検証

```sh
uv run --locked pytest tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

cosine の値・解析的勾配・補間・極小値、DSI の AIFI/backbone/projector への勾配、
DN／空ターゲット、teacher の凍結、optimizer の所有権、推論時の分離を検証します。
GAM は範囲内外・上下の制限・停止・再開を確認し、CPU の小型 student では
連続2 epoch と1 epoch + checkpoint 再開の state dict の完全一致を検証します。
`teacher=smoke` では、既定の R18vd student による synthetic 1 epoch と
Penn-Fudan 実画像1バッチの CLI 学習も確認しています。
