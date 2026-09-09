# Lightning 学習と epoch 境界での再開

[English version](../lightning.md)

`clean_rtdetr_training.lightning` は既存の DN training model 用の薄いアダプターです。Lightning の `Trainer` が automatic optimization、gradient clipping、epoch scheduling、logging、checkpoint の保存と復元を管理します。本プロジェクトは RT-DETR の loss step、optimizer parameter group、EMA、アプリケーション RNG callback state を提供します。

検証済みプロファイルは **単一プロセス CPU FP32** で、batch ごとに optimizer step を 1 回、DataLoader を 1 つ、専用 CPU generator を 1 つ使い、`num_workers=0` とします。テストは合成データを使います。COCO augmentation、完全な収束、GPU/DDP/AMP、epoch 途中の任意再開、研究手法の組み合わせは未実装です。

<a id="install-and-run"></a>

## インストールと実行

```sh
uv sync --package clean-rtdetr-training
uv run clean-rtdetr-train \
  trainer.max_epochs=1 output_dir=outputs/lightning
uv run clean-rtdetr-train \
  trainer.max_epochs=2 output_dir=outputs/lightning \
  resume=outputs/lightning/last.ckpt
```

`trainer.max_epochs` は復元済み epoch を含む最終的な目標 epoch 数です。既定 recipe は一部を freeze した R50vd backbone と合成 64×64 画像の小型 detector を使います。checkpoint や dataset はダウンロードしません。実行と再起動を確認するもので、公式学習済みモデルや精度を示しません。

Lightning は `clean-rtdetr-training` distribution が所有します。core `clean-rtdetr` のインストール・import は training stack をインストールも import もしません。Hydra は training distribution とその実験 recipe に閉じています。

<a id="hierarchical-configuration-and-dependency-injection"></a>

## 階層設定と依存性注入

アプリケーションは `clean_rtdetr_training.compose_training_config` 経由で Hydra Compose API と `hydra.utils.instantiate` を使います。設定は training distribution に同梱され、target が command-line module を import することはありません。

```python
from clean_rtdetr_training import compose_training_config

config = compose_training_config("lightning_training")
```

```text
packages/training/src/clean_rtdetr_training/configs/
  lightning_training.yaml
  criterion/set.yaml
  matcher/hungarian.yaml
  denoising/contrastive.yaml
  datamodule/synthetic.yaml
  model/
    compact.yaml
    backbone/r50vd.yaml
    encoder/compact.yaml
    projector/compact.yaml
    selector/compact.yaml
    decoder/compact.yaml
  optimizer/
    adamw.yaml
    sgd.yaml
  scheduler/
    multistep.yaml
    step.yaml
```

model group は `PResNet`、`HybridEncoder`、`MemoryProjector`、`QuerySelector`、`QueryAssembler`、`Decoder` を再帰的に instantiate し、`RTDETR` に注入します。共有次元は root config から補間され、互換性のない feature spec や次元は既存の validation が拒否します。constructor は feature spec を sequence で受け取るため YAML では通常の list を使い、各 module は内部で不変 tuple を保持します。

Lightning module は training model、criterion、parameter-group policy、optimizer factory、scheduler factory を constructor injection で受け取ります。optimizer と scheduler の config は `_partial_: true` なので、Hydra が先に factory を作り、`configure_optimizers` が実際の trainable parameter と結び付けます。adapter は固定の optimizer/scheduler class を instantiate しません。

```sh
uv run clean-rtdetr-train \
  optimizer@module.optimizer_factory=sgd \
  scheduler@module.scheduler_factory=step \
  trainer.max_epochs=1 output_dir=outputs/sgd
```

既定 AdamW recipe は固定 R50vd 設定に従い、LR `1e-4`、backbone LR `1e-5`、betas `(0.9, 0.999)`、weight decay `1e-4`、epoch ベースの `MultiStepLR(milestones=[1000], gamma=0.1)` を使います。`RTDETRParameterGroups` は注入可能な callable で、parameter name を上流名へ写像して encoder/decoder bias と `norm.*weight` の decay 除外を適用します。DN embedding を含む全 trainable parameter は optimizer にちょうど一度だけ属する必要があります。clipping は Trainer が管理し、recipe は norm limit `0.1` を設定します。別 factory は DI の例であり、上流の学習 recipe と一致することは意味しません。

<a id="model-and-criterion-interfaces"></a>

## Model と criterion の interface

adapter は `DenoisingTrainingModel` や `SetCriterion` ではなく `TrainingModel[PredictionT]` と `TrainingCriterion[PredictionT]` に依存します。どちらも abstract `nn.Module` subclass なので、通常の parameter/buffer 登録、forward hook、device 移動、state dict を維持します。generic prediction type が model と criterion を対応付け、Lightning adapter は prediction payload を調べません。

| Interface | 必須操作 |
| --- | --- |
| `TrainingModel[PredictionT]` | `forward(images, targets=...)`, `predict(images)`, `checkpoint_config()` |
| `TrainingCriterion[PredictionT]` | `forward(output, targets)`, `checkpoint_config()` |
| `Matcher` | `forward(predictions, targets)`, `checkpoint_config()` |
| `DenoisingGenerator` | `forward(targets, num_queries=...)`, dimensions/device/dtype, `checkpoint_config()` |
| `Detector` | `forward(images, denoising=..., plan=..., collect_aifi=...)`, dimensions, `checkpoint_config()` |

`SetCriterion` は抽象 `Matcher` interface の注入実装を必要とします。baseline Hydra config は `HungarianMatcher` を明示的に構築し、別実装は `_target_` で選べます。Matcher は予測 device 上の画像ごとの `MatchIndices` を返し、detection/target を検証します。criterion は matcher の `.config` 属性を要求せず `checkpoint_config()` から再開設定を得ます。

DN wrapper は `Detector` と `DenoisingGenerator` に依存します。decoder/selector config や特定の embedding layout を調べず、公開された dimensions、device、dtype、checkpoint configuration を使います。準備済み DN query は assignment metadata を含み、detector は matching と DN prediction を分離して返し `ModelOutput.final` の契約を保ちます。

`RTDETR` は `clean_rtdetr.models.interfaces` の抽象 backbone、encoder、projector、selector、assembler、decoder stage を受け取ります。各 stage は tensor 境界、互換性 metadata、checkpoint 設定を宣言します。実装は登録済み PyTorch module のままです。core は設定 framework に依存しない固定 inference baseline を `rtdetr_r50vd()` で公開し、実験 composition は training distribution の Hydra config group に残します。

```python
from clean_rtdetr import rtdetr_r50vd

model = rtdetr_r50vd()
```

training distribution は model group を training recipe とともに package します。共有 dimension は root recipe で制御し、DN generator と dataset との整合性を composition と構築時に検査します。

`criterion`、`matcher`、`denoising` config group は model-stage、optimizer、scheduler group とともに選択肢を公開します。例えば `packages/training/src/clean_rtdetr_training/configs/matcher/experimental.yaml` を追加した後、`matcher@module.criterion.matcher=experimental` で選べます。replacement config は独自の `_target_` と constructor 引数を提供し、command-line module は不要です。

`predict` は target-free 経路で、呼び出し元の train/eval mode を保ちます。criterion は重み付きで微分可能な scalar loss の辞書を返します。`checkpoint_config()` は tensor 以外の設定を primitive、sequence、mapping で返し、resume 時に実装 identity と設定を検査します。tensor state は PyTorch の `state_dict` に残ります。adapter は model の `.model`/`.denoising` や criterion の `.matcher`/`.config` に侵入しません。

注入された decoder layer と selector の score/box head も checkpoint recipe に含まれ、具体的な型と tensor 以外の設定が検査されます。組み込みの `DecoderLayer`、`BoxHead`、正確な `torch.nn.Linear` 型はこの設定を提供します。これらの型の subclass を含む custom concrete type は `checkpoint_config()` を明示的に定義し、動作に影響する全設定を辞書で返す必要があります。opaque module は通常の forward には使えますが、checkpoint recipe を要求すると `TypeError` になります。これにより、同じ shape の weight を受け入れながら dropout や head の動作が再開時に黙って変わることを防ぎます。

baseline DN wrapper と set criterion はこれらの interface を実装します。別の model/criterion pair は独自 prediction type と内部構造を使い、Hydra `_target_` で選択できます。baseline 実装を継承する必要はありません。学習可能な全パラメータをまとめる既定のグループ分けには `parameter_groups_factory: null` を使います。必要に応じて適切な方針を注入することもできます。`RTDETRParameterGroups` は RT-DETR parameter name 専用です。trainable criterion parameter も登録・最適化され、model と共有する parameter は一度だけ含めます。

interface ベースの checkpoint recipe は schema version 2 です。未リリースの以前の concrete adapter recipe の checkpoint は拒否されるため、新しい run を使ってください。

<a id="application-composition"></a>

## アプリケーション構成

アプリケーションは `datamodule` config group から `LightningDataModule` を instantiate し、`Trainer.fit(..., datamodule=...)` に渡します。既定 `DetectionDataModule` は Hydra partial の dataset/DataLoader factory を受け取り、`setup("fit")` で dataset を作り、`train_dataloader()` で loader を提供します。繰り返しの setup でも同じ dataset を保持します。collation、batching、shuffling は loader factory で設定できます。別実験では独自 `LightningDataModule` に置き換えられます。

datamodule と `EpochBoundaryState` には **同じ** generator object を注入します。datamodule はその generator を DataLoader に渡し、epoch 境界 RNG restore 契約を保ちます。entry point は composition と実行だけを行い model factory を持ちません。Hydra 1.3 の CLI help 構築が検証済み Python 3.14 argparse と非互換なため、小さな argument parser と Compose API を使います。CLI が受け付けるのは Hydra override expression で、launcher/multirun flag ではありません。

各 batch は `(images, targets)`、すなわち FP32 RGB images `[B,3,H,W]` と `DetectionTarget` の list です。[target contracts](training.md#api-and-targets) を参照してください。adapter は immutable target を明示的に移動し、全 decoder layer を保持し、matching と DN の重み付き loss を合計して `train/loss` を log します。validation と COCO metric はこの adapter とは別です。

<a id="state-and-reproducibility"></a>

## State と再現性

Lightning は model、optimizer、scheduler、epoch、global step、loop progress、callback state を通常の checkpoint 形式で保存します。追加する state は次のとおりです。

- `RTDETREMA`: DN wrapper の別個の frozen copy、update count、decay、warmup 設定。浮動小数点 parameter/buffer は upstream の `decay * (1 - exp(-updates / warmups))` に従い、整数 buffer は初期値のままです。EMA は optimizer step 後に更新し live weight を置換しません。
- `EpochBoundaryState`: Python、global NumPy、PyTorch CPU、供給された DataLoader generator の state。callback は Lightning の resume lifecycle で復元し、不完全な epoch の checkpoint を拒否します。
- Recipe metadata: factory target と引数、loss/DN/model 設定、trainable parameter name、更新に影響する Trainer 設定。不一致の resume 設定は拒否されます。再開には EMA/RNG callback も必要です。

再開には同じ dataset、batch size、sampler 設定、transform、software/runtime 環境が必要です。これらは呼び出し元の責任であり、任意の dataset object や private augmentation generator は serialize されません。worker RNG、stateful streaming dataset、world size の変更は検証対象外です。全 state を持つ完了済み epoch checkpoint を優先してください。重みだけのロードは optimizer history を引き継ぐ continuation ではなく初期化です。

baseline wrapper では live detector は `module.training_model.model` にあります。fit 後、`ema.module` は evaluation mode の別 EMA wrapper を保持します。どちらも通常の PyTorch module なので inference に Lightning Trainer は不要です。checkout 専用の `tools.compatibility.training.load_official_training_weights` は Lightning adapter 構築前に `training_model` を初期化できます。パッケージで使う場合は `training_model.state_dict()` を `torch.save` で保存し、core の native loader `clean_rtdetr.checkpoint.load_model_weights` で読み込みます。公式 envelope の converter は推論専用で、公式重みは Lightning resume checkpoint ではありません。

<a id="validation"></a>

## 検証

受け入れテストは、連続学習と epoch checkpoint から resume した新しい module/Trainer を比較します。shuffle data、確率的入力、DN noise、空 target、LR milestone を実行し、live weight、optimizer moment、scheduler progress、EMA state、step count、RNG state を比較します。厳密な restart test は新しい interpreter で行い、Python RNG state を消費する確率的 runtime type-check instrumentation は使いません。別テストは EMA 更新を固定上流実装と比較します。custom prediction payload を持つ独立 model/criterion pair でも、実際の Trainer 更新と checkpoint restore、forward hook、trainable criterion state、shared parameter、実装/設定変更の拒否を確認します。これらは CPU の restart 境界を確立しますが、学習再現や AP を示しません。

上流学習の既定値と EMA の provenance は [third_party/RT-DETR](../../third_party/RT-DETR/README.md) に記録されています。Lightning interface は [LightningModule guide](https://api.lightning.ai/docs/pytorch/stable/common/lightning_module.html) と [checkpoint guide](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/checkpointing_basic.rst) に説明があります。
