# Lightning による学習とエポック境界からの再開

[English](../lightning.md) | 日本語

`clean_rtdetr_training.lightning` は、注入した `TrainingModel` と `TrainingCriterion` を
Lightning の自動最適化に接続する。RT-DETR の学習ステップ、optimizer のパラメータグループ、
EMA、アプリケーションの乱数状態を提供する。最適化、勾配の clipping、スケジュール、ログ、
通常のチェックポイント保存は `Trainer` が担当する。
まず付属 recipe で構成とエポック境界からの再開を確認し、実際の学習には独自のデータセットと recipe を用意する。

検証済みの構成は単一プロセスの CPU `32-true`/`bf16-mixed` と CUDA `32-true`/`16-mixed` である。
DataLoader は 1 つ、`num_workers=0`、`accumulate_grad_batches=1` を使う。
このアダプターは他の精度形式、複数 rank、勾配蓄積、worker を使う厳密な再開を拒否する。
native criterion の DDP 損失正規化は別である。[GPU 検証](gpu-validation.md)を参照する。
COCO の画像拡張、収束、任意のエポック途中からの再開は対象外である。

<a id="install-and-run"></a>

## 付属 recipe で動作を確認する

リポジトリのルートから実行する。

```sh
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.max_epochs=1 output_dir=outputs/lightning
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.max_epochs=2 output_dir=outputs/lightning \
  resume=outputs/lightning/last.ckpt
```

`trainer.max_epochs` は、復元済みのエポックを含む合計の目標エポック数である。
既定 recipe は一部を凍結した小型の R50vd 構成と 128×128 の合成画像を使う。
重みやデータをダウンロードせず、実行と再開を確認するための例である。公式モデルや精度を実証するものではない。
CPU BF16 では、モデルと optimizer の状態を FP32 に保って autocast を使う。

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  trainer.precision=bf16-mixed trainer.max_epochs=1 output_dir=outputs/lightning-bf16
```

`trainer=cuda` で付属の CUDA 設定を選ぶ。CUDA の `grid_sample` の逆伝播は決定的ではないため、
この設定では `16-mixed` と `deterministic: warn` を使う。

<a id="hierarchical-configuration-and-dependency-injection"></a>
<a id="application-composition"></a>

## 構成を変更する

training の wheel に YAML は同梱されない。CLI には `--config-dir` と `--config-name` で
外部の Hydra 設定ツリーを指定し、その後に Hydra の上書き引数を渡す。
`compose_training_config(name, overrides=(), *, config_dir=...)` でも同じ構成を作れる。
相対パスは現在のディレクトリから解決する。

ルート recipe は `module`、`datamodule`、`trainer` を選ぶ。
module の recipe は学習モデル、criterion、matcher、パラメータのグループ分け、
optimizer と scheduler の factory を選ぶ。
部品を交換する場合は設定 group を上書きし、値を変える場合はドット区切りのキーを使う。

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training \
  module/optimizer_factory=sgd module/scheduler_factory=step \
  trainer.max_epochs=1 output_dir=outputs/sgd
```

既定モデルは、小型の部品を接続する例である。
`module/training_model/model=rtdetrv2_r18vd` で名前付きモデルのプリセットを選び、
`module.training_model.model.num_classes=80` でクラス数を指定できる。
プリセットはランダムな重みで初期化する。
合成データの factory はモデルを参照せず、画像サイズとクラス数を明示的に受け取る。
画像サイズは正の stride の倍数とし、projected memory の位置数が query 数以上になるようにする。

optimizer と scheduler の設定は Hydra の partial factory である。
アダプターは `configure_optimizers` で実際のパラメータを渡し、DN や学習可能な criterion を含む
すべての学習可能パラメータを一度ずつ所有していることを確認する。
既定 recipe は AdamW（`lr=1e-4`、backbone `1e-5`、betas `(0.9, 0.999)`、weight decay `1e-4`）、
エポック単位の `MultiStepLR`、ノルムによる勾配 clipping `0.1` を使う。
これは基本構成の例であり、学習全体の再現を実証した設定ではない。

<a id="model-and-criterion-interfaces"></a>

## モデルと criterion の契約

アダプターは、固定のモデル構造ではなく次のインターフェースに依存する。

| インターフェース | 必要な操作 |
| --- | --- |
| `TrainingModel[PredictionT]` | `forward(images, targets=...)`、`predict(images)`、`checkpoint_config()` |
| `TrainingCriterion[PredictionT]` | `forward(output, targets)`、`checkpoint_config()` |
| `Matcher` | `forward(predictions, targets)`、`checkpoint_config()` |
| `DenoisingGenerator` | `forward(targets, num_queries=...)`、次元・device・dtype、`checkpoint_config()` |

`forward` は対応する criterion が受け取る payload を返す。
`predict` はターゲットなしの推論経路で、呼び出し前の train/eval 状態を保つ。
criterion は重み付きで微分可能なスカラー項を返し、その合計を最適化する。
標準の `DenoisingTrainingModel` と `SetCriterion` は `TrainingOutput` を使い、
その `.final` には matching query の予測だけを含める。
バッチは FP32 RGB 画像 `[B, 3, H, W]` と、画像ごとに 1 つの `DetectionTarget` である。
[ターゲットの契約](training.md#api-and-targets)を参照する。

`checkpoint_config()` では、学習継続に影響する非テンソル設定を基本型・シーケンス・マッピングで記述する。
テンソルの状態は通常の PyTorch state dict に保存する。
挙動に影響する独自 module もこのメソッドを実装する。
設定を記述できない module は通常の実行には使えるが、再開用の構成を要求すると失敗する。
チェックポイントの schema は version 2 で、以前の未リリース版のアダプターのチェックポイントは拒否する。

<a id="state-and-resume"></a>
<a id="state-and-reproducibility"></a>

## 状態を保存して再開する

Lightning はモデル、optimizer、scheduler、epoch、global step、ループの状態、callback の状態を保存する。
付属 callback とメタデータは次を追加する。

- `RTDETREMA`: 凍結した `training_model` のコピー、更新回数、減衰設定。
  optimizer step が成功した後だけ更新し、AMP overflow によるスキップでは更新しない。
- `EpochBoundaryState`: Python、NumPy、PyTorch CPU、使用中の CUDA device の RNG と、
  共有する DataLoader generator の状態。エポック途中のチェックポイントを拒否する。
- recipe と実行のメタデータ: 部品の設定、学習可能なパラメータ名、精度形式、device の種類、
  clipping、勾配蓄積。設定を変えた再開は拒否し、必須 callback の存在も確認する。

厳密なエポック境界からの再開には、データセット、バッチ分割、sampler、変換、環境を揃え、
付属の CPU generator を使う DataLoader を 1 つだけ使う必要がある。
これらのデータ側の条件は保存されない。状態を持つデータセットやストリーミング、worker の RNG、
world size の変更、任意のエポック途中からの再開は対象外である。
モデルの重みだけを読む場合は新しい実行の初期化になり、optimizer の履歴は復元しない。

学習後の `module.training_model.model` は学習中に更新した検出器、`ema.module` は別の評価モードの wrapper である。
予測・エクスポート CLI で使う場合は `training_model.model.state_dict()` を保存する。
公式推論重みの変換コマンドは、Lightning の再開用チェックポイントを生成しない。

<a id="validation"></a>

## 検証範囲

受け入れテストでは、連続学習と、完了したエポックのチェックポイントから新しい module・Trainer で
再開した学習を比較する。シャッフルした合成データ、確率的な入力、DN ノイズ、空ターゲット、
学習率の milestone、CPU FP32/BF16 と CUDA FP32/AMP を対象とする。
学習中の重み、optimizer と scheduler の状態、EMA、step、RNG を比較し、
CUDA の継続比較には文書化された許容誤差を使う。
全データセットの学習や複数 rank の Lightning を検証したものではない。
