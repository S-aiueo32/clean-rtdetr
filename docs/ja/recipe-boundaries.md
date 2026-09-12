# Recipe を通じて整理した責務

[English version](../recipe-boundaries.md)

Clean RT-DETR は、読みやすい PyTorch の参照実装である。4つの recipe は拡張点の
利用例であり、core を学習フレームワークにするためのものではない。
このガイドでは、コードを package と application のどちらに置くかを判断し、recipe を
現在の training contract へ移行する方法を説明する。

| 所有者 | 責務 | 例 |
| --- | --- | --- |
| `clean_rtdetr` | モデルの tensor 計算、次元、観測可能な stage 出力 | 特徴、query 組み立て、attention mask、decoder 予測、AIFI パラメータ |
| `clean_rtdetr_training` | 標準 RT-DETR の教師との対応付けと検証済み adapter | target、matching、DN 割り当て、VFL/L1/GIoU、単一プロセス正規化、epoch 境界再開 |
| recipe / application | 研究手法と実験の構成 | MAL、FDR、SAP/O2M、DSI/GAM、データ設定、teacher 重み、augmentation、schedule |

recipe 間で共有するコードはレビュー候補であるが、自動的に公開 API にはならない。
API は package 内で安定した contract を表し、recipe の種類を知らずに動作し、その
recipe とは独立して検証できる必要がある。手法固有の flag、使われない設定、
wrapper の内部探索、実験用の特例は recipe に置く。

## Core の query と training の教師割り当て

core の `DenoisingQueries` は準備済み content、reference logits、attention mask
だけを持つ。`QueryAssembler` は tensor の連結、形状検証、初期 reference の
detach を担当する。これらはモデルの計算であり、target index、DN group、
loss の割り当てを必要としない。

`RTDETR.forward(denoising=...)` は準備済み query を受け取り、DN を先頭に含む
全 decoder slot を返す。`ModelOutput` の field は `decoder`、`encoder_aux`、
`after_aifi` だけである。`final` も全 slot を含み、教師の付け方を判断しない。

`DenoisingMetadata`、`DenoisingBatch`、`DenoisingOutput`、`TrainingOutput`、
`split_denoising_predictions` は training package が所有する。
`DenoisingTrainingModel` は `prepared.queries` だけを detector に渡す。
戻り値の `TrainingOutput.predictions` は matching query だけの `ModelOutput`、
`denoising` は DN 予測と教師割り当て、`final` は matching query だけである。
DN loss も計算するには出力全体を criterion に渡す。

```python
output = training_model(images, targets=targets)
losses = criterion(output, targets)
matching_predictions = output.final
aifi_features = output.predictions.after_aifi
```

公開 split helper は、保持した全 layer と勾配を維持する。DN がある場合は
標準 `DecoderOutput` だけを対象とし、追加 tensor を持つ独自出力は recipe 自身で
分割する。DN がなければ元の object をそのまま返し、独自 payload を維持する。
RT-DETRv3 は標準分割を各分岐で再利用し、mask、教師の繰り返し、分岐の集約を
recipe で実装する。

## Loss の共通処理と目的関数の所有者

`SetCriterion` は参照実装の VFL/L1/GIoU に限定する。`CriterionConfig` には
その重みと Varifocal の設定だけを持たせる。分類式を変える場合は別の criterion
を実装する。旧 `ClassificationLoss` hook と `SetCriterion` の注入引数は削除し、
MAL を使う checkpoint 設定に無効な VFL 項目が残る構成を解消した。

`SetCriterion.forward` は training model と同じ契約の `TrainingOutput` を受け取る。
core の `ModelOutput` は明示的に `TrainingOutput` で包み、単一 stage の
`DetectionOutput` には `criterion.stage_losses(detection_output, targets)` を使う。
公開 `match_stage` helper は、criterion の入力型を広げずに単一 stage の割り当てと
正規化係数を再利用するための関数である。

共通処理は `clean_rtdetr_training.losses` の明示的な関数として公開する。

| 関数 | 契約 |
| --- | --- |
| `iter_matched_stages` | final/auxiliary の独立 matching、標準 DN 割り当て、単一プロセスの正規化係数 |
| `gather_matched_pairs` | 対応した予測・教師 box と label の取得。予測の勾配を維持 |
| `matched_iou_targets` | detach した IoU 教師と、IoU=0 の正例も保持する別 mask |
| `matched_box_losses` | 指定した正例数で正規化する、重み適用前の L1/GIoU |

これらは標準 RT-DETR の教師の付け方を実装する。分類式の選択、重みの所有、
recipe の種類による分岐は行わない。DEIM の `MALCriterion` は関数を明示的に
組み合わせ、`gamma`、`weight_mal`、`weight_bbox`、`weight_giou` だけを持つ
`MALConfig` を所有する。別の assignment、stage 集約、分散正規化を使う場合は
その criterion で実装する。D-FINE の union matching と分布 loss も recipe に残する。

`HungarianMatcher(prediction_box_range="unbounded")` は training に残する。
正規化座標系であることと、予測が画像内に収まることは別の契約だからである。
既定の `"unit"` は従来どおり予測の `[0, 1]` を検証する。`"unbounded"` では
範囲外の有限な cxcywh 予測を受け入れ、幅・高さの非負性を検証する。
target の正規化検証は維持する。D-FINE は cost と solver を複製せずにこの方針を
選べる。core への分布回帰の追加や export 対応の拡張ではない。

## データの構成は application が決める

dataset と `DetectionDataModule` は training model を調べない。CLI が注入するのは
共有 DataLoader generator である。datamodule は設定を束縛済みの引数なし dataset
factory を受け取る。データ構築のために `TrainingModel.num_classes` や
wrapper の `.model` を要求しない。

公開 Penn-Fudan loader はファイル・mask の読み込みと `person=0` のラベル契約を
所有する。対応する検出 head、split、seed は recipe が選ぶ。
synthetic loader は明示した `num_classes` と `image_size` を受け取る。

```yaml
dataset_factory:
  _target_: clean_rtdetr_training.data.SyntheticDetectionDataset
  _partial_: true
  num_classes: 3
  image_size: 128
```

同じ設定値を model と dataset に補間することは、application の明示的な構成である。
構築済み model object を渡す必要はない。stride に合う入力サイズと、
query 数に足りる projected memory の位置数は caller が確保する。
`image_size=auto` と `from_training_model` constructor は削除した。
root の synthetic、DEIM、RT-DETRv4 は128、D-FINE、RT-DETRv3 は160を指定する。
特に v3 の450-query分岐には、128入力で得られる336位置では足りない。

実データの augmentation、category 対応、検証指標、private RNG、batching、sampler
は application の判断である。既存 loader の契約が合えば再利用し、recipe に
合わせるための model 依存を loader に足さないようにする。

## 観測可能な構造と callback の状態

`HybridEncoder.aifi_parameters()` は既存 stage のパラメータを公開する。
`after_aifi` と合わせて、内部属性名に依存せずモデルを観測するための API である。
core は勾配比を計算せず、teacher、epoch、optimizer、蒸留重みを知らない。
RT-DETRv4 が DSI/GAM の全計算を所有し、この構造 API で対象パラメータを選ぶ。

オプションの Lightning adapter には `required_callback_keys` を残する。
key は再開に必要な callback を識別する値として扱い、その存在を確認する。
保存時の要求は、再開先が宣言を省いても維持する。adapter は GAM という手法を
認識せず、そのアルゴリズムを再構築しない。状態を持つ callback 自身が状態と
設定照合を所有し、recipe が key を宣言して callback の欠落した再開を防ぎる。

## 各 recipe に残すもの

| recipe | ユーザー側の責務 |
| --- | --- |
| DEIM | MAL の式、目的関数の設定、実験 schedule |
| D-FINE | FDR decoder/output、bin、LQE、FGL、union assignment、DDF |
| RT-DETRv3 | SAP mask/selector、O2M の教師繰り返し、CNN 補助 head、分岐の重み |
| RT-DETRv4 | teacher 読み込み・前処理、DSI の投影・整列、GAM の観測・更新・callback state |

4つとも checkout 内の独立したプロジェクトとして維持する。独自出力や checkpoint
が既存の推論・export CLI にそのまま対応するわけではない。公式推論重みの
変換は core の `clean-rtdetr-convert` CLI の private 実装として同梱する。
公式学習重みの互換処理と stage parity helper は `tools/compatibility` に置く
checkout 専用処理である。

## API 移行と検証範囲

- DN metadata、batch、output、split helper は `clean_rtdetr_training` から import
  する。準備済み `DenoisingQueries` は core に残る。
- decoder/encoder/AIFI の参照は `training_output.predictions` を経由する。
  DN loss の計算には `TrainingOutput` 全体を criterion に渡す。
- core `RTDETR` に渡すのは query tensor である。戻り値は全 query slot を含む。
- dataset/datamodule は設定と factory で直接構築する。データ構築のためだけに
  wrapper に足した model metadata property は不要である。
- 別の目的関数は `TrainingCriterion` で構成する。旧 `ClassificationLoss` 注入
  API の互換 shim は設けない。

model/DN の parameter state dict key と標準 `SetCriterion.checkpoint_config()` の
schema は維持する。一方で import、出力型、constructor signature は変更する。
独自 criterion やデータの設定を変えた場合の学習継続を一括保証するものではなく、
同じ完全な設定と検証済み epoch 境界再開の契約を使う。

パッケージの依存方向、全 query の出力と勾配、DN 分割、標準損失の一致、モデルに
依存しない CLI のデータ構築、recipe の目的関数、callback の再開をテストする。
この小規模な CPU 検証は、recipe の本格学習、収束、COCO AP、recipe 固有の
GPU/DDP/AMP 実行、独自 recipe のエクスポートを実証するものではない。
baseline の GPU 検証には個別の[検証範囲](gpu-validation.md)がある。
実行方法は[学習](training.md)、[Lightning](lightning.md)、各 recipe の README を参照する。
