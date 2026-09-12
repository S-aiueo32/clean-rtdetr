# GPU 実行と検証

[English](../gpu-validation.md) | 日本語

Clean RT-DETR の CUDA・混合精度に関する検証範囲を示す。
実行構成を選び、記録された GPU 検証の結果を解釈するためのページである。
学習の収束、データセット AP、レイテンシ、TensorRT 対応、精度形式間の数値一致は実証していない。

## 精度形式による挙動

検出器は FP16 の重みと CUDA FP16 autocast を受け付ける。
位置埋め込みは autocast 中も上流と同じ FP32 演算を保つ。
特徴が FP16/BF16 の場合、deformable attention の補間と box 座標の反復更新は FP32 を使う。
公開する logits と box も FP32 を保つ（FP64 実行では FP64）。
そのため、`SetCriterion` と DN 損失の計算前に呼び出し側で型変換する必要はない。

精度形式によって top-k のスコアと選択 query が変わる場合がある。
有限値の出力は実行できた根拠であり、数値一致の根拠にはならない。
比較時は selector の `source_indices` を記録し、両方が同じ ID を選んだ場合は query ID で行を対応付ける。
ID が異なる場合は、選択の重複と共通部分の数値差を別々に報告する。
共通部分の一致は出力全体の一致を示さない。

混合精度で学習する場合はモデルの重みを FP32 に保ち、autocast と
`torch.amp.GradScaler("cuda")` を使う。scaler が倍率を下げている間は、
バッチを処理してもパラメータを更新しない場合がある。エポック単位の scheduler は完了したエポックに従う。
FP16 の重みによる推論を検証しても、FP16 の optimizer による学習を検証したことにはならない。

## Lightning の CUDA 設定と DDP の境界

付属の Lightning アダプターは単一プロセスに対応する。
CPU は `32-true`/`bf16-mixed`、CUDA は `32-true`/`16-mixed` を使う。
[Lightning のセットアップ](lightning.md#install-and-run)後、次の設定で CUDA を選べる。

```sh
uv run clean-rtdetr-train \
  --config-dir configs --config-name lightning_training trainer=cuda
```

CUDA 設定では `deterministic: warn` を使う。CUDA の `grid_sample` の逆伝播は、
勾配の加算順序が決定的になることを保証しないためである。
完了したエポックのチェックポイントには、CUDA/CPU の RNG、DataLoader の generator、
optimizer、scheduler、AMP scaler、EMA を保存する。
AMP での再開はビット単位の一致ではなく許容誤差で比較する。
再開に必要な条件は[状態と再開](lightning.md#state-and-resume)を参照する。

native PyTorch DDP の損失計算は別途対応している。
ターゲット数を `all_reduce(sum) / world_size` で集約し、下限を 1 とした平均値を正規化係数に使う。
全 rank が同じ順序で criterion を呼ぶ必要がある。
この検証は、Lightning アダプターの複数 rank、勾配蓄積、worker を使う厳密な再開には適用できない。

## 記録済みの検証

[Kaggle の記録](../../benchmarks/gpu-validation.json)には、Tesla T4 2 枚、Python 3.13.13、
PyTorch 2.13.0+cu126、ソースの識別情報、シード、結果を保存している。
ランダム重みの R50vd 全体による FP16/autocast 推論と有限な FP32 出力、小型モデルでの
DN・補助損失を含む native AMP 学習、2-rank NCCL の FP32/AMP 更新、
ターゲット数が異なる場合や空の場合の全 rank による正規化、Lightning FP32/AMP の
エポック境界からの再開を確認している。いずれも合成データによる短い実行である。

[T4 の精度再検証](../../benchmarks/t4-precision-recheck.json)には、CPU/CUDA の精度形式と
AMP に関する回帰テスト 48 件の成功を記録している。
有限値で実行できることと、精度形式間の数値一致を区別している。
公式 EMA 重みと 1 つの 640×640 合成入力では、FP32・AMP のそれぞれで
Clean RT-DETR と固定した上流実装の結果が完全一致しました。
この単一入力の結果は、混合精度でのデータセット精度を示さない。

T4 は native CUDA BF16 対応を報告しなかった。
別の[過去の L4 BF16 検証](../../benchmarks/bf16-l4.md)は、記録されたソース時点の
小型合成モデルで native CUDA BF16 実行を確認したものである。
後続の T4 実装の検証には適用できず、Lightning アダプターも CUDA `bf16-mixed` を受け付けない。

## 対象を絞って再検証する

リポジトリのルートで、ワークスペース全体を準備する。

```sh
uv sync --all-packages
```

以下には CPU と 2-rank Gloo のテストも含まれる。
CUDA がない環境でスキップされるのは CUDA 固有のケースだけである。

```sh
uv run pytest packages/core/tests/test_precision.py \
  packages/core/tests/test_encoder.py packages/core/tests/test_stages.py \
  packages/training/tests/test_gpu_precision.py \
  packages/training/tests/test_cuda_amp.py \
  packages/training/tests/test_gpu_lightning.py \
  packages/training/tests/test_distributed_losses.py
```

macOS で 2-rank Gloo テストを実行する場合は、worker がローカルソケットに bind するため
`GLOO_SOCKET_IFNAME=lo0` を設定する。
CUDA scaler の回帰テストは、記録された各初期倍率について、24 回以内の試行で 3 回の
パラメータ更新を要求する。スキップした試行ではパラメータを変えず、倍率を下げる必要がある。
scaler の調整だけで終わる短い実行では、学習による更新を確認したことにはならない。

この検証は合成データと限定された実行を対象とする。
学習全体の再現、混合精度の COCO AP、任意入力での精度形式間の一致、レイテンシ、
動的形状 ONNX、TensorRT は検証していない。
