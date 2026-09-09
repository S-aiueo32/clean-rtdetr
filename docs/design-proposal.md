# Clean RT-DETR

[日本語ドキュメント一覧](ja/README.md)

Readable Reference Implementation, Research Extensions & Optimization Playground

RT-DETR の意味論を保ったまま、研究・推論変換・最適化を分離する設計案

**文書種別:** Architecture / Design Proposal

**対象:** PyTorch reference implementation、研究拡張・再現実験、export、推論最適化

**版:** v0.3 / 2026-09-05（設計提案・未実装）

**位置付け:** CleanRL / nanoGPT for RT-DETR

> **設計判断:** RT-DETR を一枚岩のモデルとして書き直すのではなく、Reference implementation → Transformation passes → Deployment backends の三層に分ける。

> **研究拡張:** Reference 層を `ModelVariant × TrainingMethod × ExperimentRecipe` の明示的な composition とする。RT-DETR の互換 baseline を保ち、RT-DETRv3 / v4、DEIM、D-FINE に必要な学習・推論の差分を注入できる境界を初版から検証する。

## 0. エグゼクティブサマリー

本プロジェクトは、RT-DETR の読みやすい参照実装を作り、公式 checkpoint との互換性を検証したうえで、推論向けの変換・最適化を重ねる基盤を目指す。設計の中心は、論文との対応を追える計算経路、明示されたテンソル境界、変更前後を比較できる検証手順の三点である。

- Reference は PyTorch と標準ライブラリを基本とし、学習フレームワーク、CLI、データセット、backend SDK に依存しない。
- 各境界に tensor contract を置き、座標系、shape、dtype、mask、勾配の扱いを明記する。
- 高速化は transformation pass と deployment profile で表現し、等価変換と精度・速度のトレードオフを区別する。
- 互換性は checkpoint、stage 出力、勾配、AP の順に検証し、レイテンシは条件を固定して計測する。
- 研究拡張は model、学習分岐、matcher / loss、データ・学習 schedule の単位で構成し、論文再現と新しい組合せの ablation を同じ基盤で実行する。

初版の互換基準は `rtdetr_pytorch` の RT-DETR-R50vd、COCO 80 classes、640×640 入力、300 queries、6 decoder layers とする。同時に、学習分岐・特徴蒸留・分布回帰・recipe 差し替えの小さな実証を行い、研究拡張を受け入れる構造を確かめる。各派生手法の完全再現と deployment 対応は段階的に追加し、研究利用の開始条件を全 backend の完成に置かない。

本書の API、閾値、対応 backend は設計案であり、実装済み機能や測定済み性能を示すものではない。

## 1. 背景と課題

現行の `rtdetr_pytorch` では、最上位の `RTDETR.forward` 自体は短い。一方、`RTDETRTransformer` に projection、anchor 生成、query selection、DN 準備、decoder、出力整形が集まり、学習・推論・export の都合が接続部分に入り込んでいる。分離すべき対象は、この責務と暗黙の入出力規約である。

| **課題**            | **現在の症状**                                           | **設計上の帰結**                                      |
|---------------------|----------------------------------------------------------|-------------------------------------------------------|
| 動的 DI / YAML      | constructor の実引数と依存関係が局所的に読めない         | composition root だけに限定し、typed config へ置換    |
| decoder の責務過多  | projection、anchor、top-k、DN、decode、aux 出力が密結合  | MemoryProjector、QuerySelector、Decoder と学習 adapter へ分割 |
| 暗黙 dict protocol  | `aux_outputs` / `dn_meta` の有無で loss 経路が変わる     | 固定フィールドと明示的な出力 policy を導入            |
| shape・座標系が無型 | normalized cxcywh、logit、flattened memory が全て Tensor | 型名・validator・境界テストで意味を固定               |
| export 都合の漏出   | ONNX 回避コードが reference forward に混在               | backend adapter と rewrite pass に隔離                |
| 重複 config         | channels、strides、levels を複数箇所で手動同期           | FeatureSpec から導出し、一つの source of truth に統合 |

確認対象は upstream `lyuwenyu/RT-DETR` の commit `068dfde65f2667ad6555883c69d73de886518cad`。設計の根拠となる箇所は次のとおり。

- [モデル構成と multi-scale resize](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/rtdetr.py)：モデル内で学習時の入力 resize と deploy 変換を実行する。
- [query selection と decoder](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py)：top-k、detach、反復 box 更新、train/eval の出力分岐を持つ。
- [HybridEncoder](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py)／[attention の補助演算](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/utils.py)：RepVGG block の deploy 変換、位置埋め込み、deformable attention の sampling 処理を確認する。
- [DN 構築](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/denoising.py)／[criterion](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py)：DN metadata、auxiliary loss、分散時の正規化に依存する。
- [postprocessor](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py)／[ONNX exporter](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/tools/export_onnx.py)：deploy 時に出力形式が変わり、現行 export は postprocess を含む。
- [基準モデル設定](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml)／[学習・評価の前処理](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/rtdetr_pytorch/configs/rtdetr/include/dataloader.yml)：初版の構成と入力規約の出発点とする。

## 2. 目標と非目標

### 2.1 目標

- 論文の各処理とコードの対応を追える reference implementation を作る。
- 公式 checkpoint を変換でき、代表入力に対して stage-wise に数値整合する。
- query selection、attention、decoder layer 数、precision、fusion を独立に交換・最適化できる。
- whole-model と memory 境界の分割 export を成立させ、後続で QuerySelector、Decoder の独立 export に広げる。
- 新規研究実装と本番推論実装が、共通 contract と golden test を共有する。
- 本リポジトリを基盤に、RT-DETRv3 / v4、DEIM + D-FINE などの構成要素を明示的に注入し、単独・組合せの効果を比較できる。
- 研究者が外部 Python package や `experiments/` から拡張を渡せるようにし、共有 runtime や baseline の forward の編集を新手法追加の前提にしない。

### 2.2 非目標

- 初版から全 RT-DETR 派生モデル・全 backbone・全データセットを網羅しない。
- 独自の巨大 training framework や汎用 object detection framework を新設しない。
- 読みやすさのために公式挙動や精度を暗黙に変更しない。改善は別 pass / variant として扱う。
- 特定 backend の最速記録だけを目的に reference semantics を崩さない。
- 全ての拡張の任意の組合せを保証しない。必要な出力・表現・学習状態の不一致は構成時に検出し、新規組合せは独立した実験として検証する。

### 2.3 初版の対応範囲

| 項目 | 初版で検証する範囲 | 後続で拡張する範囲 |
|------|--------------------|--------------------|
| モデル | RT-DETR-R50vd の互換 baseline と component 注入 | RT-DETRv2 / v3 / v4、D-FINE の完全再現、別 backbone / class count |
| 研究拡張 | 学習分岐、特徴取得、分布回帰、学習 recipe の最小実証 | 論文ごとの正式 recipe / checkpoint、DEIM + D-FINE、拡張間の組合せ |
| 入力 | 評価時は RGB、640×640、padding なし。学習は固定した公式 recipe に従う | 可変 H/W の推論・export、letterbox、padding mask |
| 重み | 公式配布の PyTorch 形式 checkpoint からの変換 | Paddle 形式の直接読込、他 variant |
| 実行 | FP32 eager、固定 shape の ONNX Runtime。後段で TensorRT FP16 | OpenVINO、Torch-TensorRT、BF16、INT8 |
| export | whole-model を先行し、memory 境界での分割を追加 | 全コンポーネントの独立配布、可変 shape |

「初版」は第12節の Phase 0〜5 を通じた到達点とする。研究利用の最初の到達点は Phase 3 と研究拡張の最小実証とし、TensorRT の完成や全派生モデルの AP 再現を待たずに実験を始められる状態を作る。

## 3. 設計原則

| **原則**                   | **内容**                                                                             |
|----------------------------|--------------------------------------------------------------------------------------|
| P1: Semantics first        | reference path は論文と公式実装の意味論を最優先し、高速化を混ぜない。                |
| P2: Explicit contracts     | Tensor の shape、layout、座標系、値域、dtype、gradient policy を境界ごとに定義する。 |
| P3: Explicit execution    | BN / Dropout の train/eval 意味論は維持し、DN、出力収集、実行 layer 数を明示する。   |
| P4: One-way dependency     | core ← transformations ← backends の依存方向を守り、core は backend を知らない。     |
| P5: Optimization as a pass | 融合、事前計算、precision 変更、kernel 置換は宣言的 pass と検証結果を持つ。          |
| P6: Parity is a feature    | baseline と検証済み profile は精度・checkpoint・stage / export の gate を持つ。研究拡張は宣言した検証段階に応じて判定する。 |
| P7: Explicit composition | model variant、training method、recipe を constructor / factory で注入し、依存と parameter の所有者を追えるようにする。 |
| P8: Variant-local semantics | 共通の最終検出形式を維持し、内部の回帰表現・query group・勾配規約は variant ごとに定義する。 |
| P9: Research before deployment | contract / gradient 検証を通した研究拡張を受け入れ、再現済み・export 済み・backend 検証済みを別々に表示する。 |

## 4. 全体アーキテクチャ

```text
User API / CLI / Examples
        |
        v
Composition: ModelVariant + TrainingMethod + ExperimentRecipe
        |
        v
Reference implementation
  images -> Backbone -> HybridEncoder -> MemoryProjector -> memory
  memory -> QuerySelector -> matching queries
  matching queries + DN queries (training only) -> QueryAssembler -> queries
  memory + queries -> Decoder (layer + prediction heads) -> ModelOutput
  ModelOutput.final -> Postprocessor -> detections

  reference module + weights + profile
                  |
                  v
Transformation passes
  fusion | shape specialization | precision | attention replacement | layer selection
                  |
                  v
Deployment backends
  Tensor-only export (torch.export / ONNX)
    -> ONNX Runtime | TensorRT | future backends
  Transformed module -> PyTorch runtime
```

図の矢印は計算・変換の流れを示す。コードの依存は `backends → transforms / export → models / core` の一方向とする。変換層は reference module とその重みを受け取り、contract、profile、検証 fixture を使って deployment 用 module を作る。初版では独自の汎用 IR を新設せず、PyTorch module と Tensor-only export schema を使う。backend 固有 object や plugin handle は core に逆流させない。

図中の query / decoder 経路は RT-DETR baseline の例である。研究拡張は三層に並ぶ第四の変換層ではなく、Reference 層を組み立てる仕組みとする。`variants/` と `methods/` は共通 contract に依存し、`core/` と baseline は個別の論文名を知らない。

### 4.1 Reference implementation

| **コンポーネント** | **責務**                                      | **責務外**                   |
|--------------------|-----------------------------------------------|------------------------------|
| Backbone           | 画像から multi-scale feature を生成           | query、box、postprocess      |
| HybridEncoder      | intra-scale interaction と cross-scale fusion | top-k、decoder state         |
| MemoryProjector    | decoder 用 projection、追加 level、flatten   | encoder 内の projection の置換 |
| QuerySelector      | anchor、encoder score / box、top-k、query 初期化 | decoder layer loop、DN 構築 |
| DenoisingBuilder   | 学習時の DN content、box logits、mask、metadata を生成 | 推論、criterion |
| QueryAssembler     | DN と通常 query の連結、参照 box の detach    | ノイズ生成、loss 計算         |
| Decoder            | attention、反復 box 更新、layer 別予測の収集 | top-k 選択、画像座標復元      |
| PredictionHeads    | layer 別 class logits / box delta の計算。Decoder が所有 | 出力の API 整形、NMS |
| Postprocessor      | score 化、top-k、座標復元、label mapping      | training graph               |

各責務はまず少数の module と補助関数で表現する。表の行ごとに公開 interface や class を増やす必要はない。特に box head は次 layer の reference box を更新するため、decoder の後段に一括で移すことはできない。拡張時は Decoder が head と refinement state の整合を担い、必要に応じて decoder 全体を交換する。

### 4.2 中間表現（tensor contracts）

`B` は batch、`N` は feature level 数、`C` は hidden dimension、`S=ΣHᵢWᵢ` は memory 長、`Q` は通常 query 数、`D` は DN query 数、`L` は実行 layer 数、`K` は class 数を表す。

以下は RT-DETR baseline の契約である。全 variant に共通とするのは feature 境界と最終 `DetectionOutput` を中心とし、`ref_logits`、補助出力数、DN 構造を全手法へ強制しない。

| **型**             | **主要フィールド**                           | **不変条件**                                    |
|--------------------|----------------------------------------------|-------------------------------------------------|
| ImageBatch         | `images[B,3,H,W]`、`original_sizes_wh[B,2]` | RGB / NCHW、初版の評価入力は float `[0,1]`、mean/std 正規化なし。model は images のみを受け取る |
| MultiScaleFeatures | `levels: tuple[Tensor, ...]`、specs         | level は stride 昇順、各 Tensor は `[B,Cᵢ,Hᵢ,Wᵢ]` |
| EncodedFeatures    | `memory[B,S,C]`、`spatial_shapes[N,2]`、`level_start_index[N]` | shapes は `(Hᵢ,Wᵢ)`、index は prefix sum、整数 Tensor は int64 |
| QuerySet           | `content[B,Q,C]`、`ref_logits[B,Q,4]`、`source_indices[B,Q]` | ref は normalized cxcywh の logit。選択 index は int64 |
| SelectionOutput    | queries、`encoder_logits[B,Q,K]`、`encoder_boxes[B,Q,4]` | encoder 補助損失用の予測は勾配を保持 |
| DenoisingInput     | `content[B,D,C]`、`ref_logits[B,D,4]`、attention mask、metadata | `D` は batch に依存。DN 無効時は空 Tensor / 空 metadata |
| DetectionOutput    | `logits[B,Q,K]`、`boxes[B,Q,4]`              | logits は sigmoid 前、boxes は normalized cxcywh、postprocess 前 |
| ModelOutput        | final、decoder_aux、encoder_aux、dn_outputs、research | baseline の research は EmptyResearchOutput。拡張時は variant が定義する型付き payload を使う |

境界には次の規約も含める。

- **Flatten:** level ごとに `feature.flatten(2).transpose(1, 2)` とし、W が先に進む順で連結する。feature の実 shape を正とし、入力サイズと stride だけから丸めを推測しない。
- **座標:** box の成分順は `(cx, cy, w, h)`、画像サイズは `original_sizes_wh=(W,H)` とする。初版の評価では画像全体を 640×640 に resize するため、normalized box を元画像の W/H で復元できる。letterbox を追加する際は scale / padding の逆変換を別 contract に含める。`ref_logits` と sigmoid 後の box を型名で区別し、`inverse_sigmoid` の clamp / epsilon は基準実装に合わせる。`xyxy` 変換後に画像外へ出る box を暗黙に clip しない。
- **Mask:** anchor の有効性は `[1,S,1]` の bool で `True=valid`、self-attention mask は `[D+Q,D+Q]` の bool で `True=遮断` とする。両者を同じ `valid` として扱わない。padding mask は初版の対象外。
- **Dtype / device:** FP32 baseline を基準とし、整数・bool を除く演算 dtype と必要な昇格を記録する。低精度化は profile で指定する。無効 anchor の `+inf` は意図した表現として比較し、NaN や想定外の非有限値と区別する。
- **出力数:** 通常 query の `Q` と DN の `D` を別軸として管理する。`decoder_aux` は final を除く layer の予測を実行順に保持し、`encoder_aux` は top-k の encoder 予測、`dn_outputs` は final layer を含む DN 予測を保持する。DN の logits / boxes はそれぞれ `[B,D,K]` / `[B,D,4]` とする。

dataclass は可読性用の Python wrapper とし、export boundary では schema version と名前・順序を固定した tuple of tensors に lower する。「固定 schema」は固定フィールドを意味し、全 layer 分の空ダミー Tensor を毎回返すことではない。export 時は policy を固定して出力数を確定する。

`ModelOutput[ResearchPayload]` は概念上の generic とし、payload 型は variant 構築時に決める。例えば D-FINE の分布出力を RT-DETR の空フィールドとして常設しない。同じ variant の train/eval では payload のフィールドを固定し、収集しない内容はその型の空表現で返す。学習 method は payload と明示的な `FeatureTaps` から `TrainingResult` を作り、共通 postprocess は `final` だけに依存する。

### 4.3 RT-DETR baseline の互換性を守る計算上の要点

- Query selection は基準実装の `enc_logits.max(-1)` に対する top-k を保存する。anchor mask による memory のゼロ化を、score の `-inf` masking へ暗黙に置き換えない。
- 通常 query の content は、feature 初期化時には gather 後に detach する。learned query を使う variant では embedding への勾配を保持する。encoder 補助予測は detach しない。
- DN を連結した初期 reference logits は detach する。学習時の layer 間 reference box の detach と、補助 box 予測に使う勾配付き reference の経路を区別する。前向き出力の一致だけではこの互換性を保証できない。
- Decoder の self-attention と multi-scale deformable cross-attention は別の交換点にする。cross-attention を通常の SDPA と同じ演算として扱わない。
- 初版の postprocess は sigmoid 後の `Q×K` 全体から top-k を選ぶ。query ごとの最大 class 選択とは異なり、同じ box が複数 label で残り得る。NMS や score threshold は公式 AP 比較の標準経路に追加しない。

### 4.4 代表的な研究拡張から導く要件

以下の「手法の変更点」は一次資料に基づく要約であり、「必要な注入点」は本プロジェクトの設計判断である。確認日は 2026-09-05。資料の参照は実装互換性の保証を意味せず、移植時に source commit、config、checkpoint を固定する。

| 手法 | 手法の変更点と一次資料 | 必要な注入点 |
|------|------------------------|--------------|
| RT-DETRv3 | CNN 補助 head、self-attention perturbation を使う複数 query group、重み共有 decoder の one-to-many 分岐を学習時に追加する。[論文](https://arxiv.org/abs/2409.08475)・[公式実装](https://github.com/clxia12/RT-DETRv3) | encoder feature の取得、学習分岐の実行、query / mask の構成、共有 decoder 呼出し、分岐別 assignment / loss |
| RT-DETRv4 | VFM の特徴を AIFI 出力へ蒸留する DSI と、勾配比率を使って重みを調整する GAM を学習時に用いる。[論文・手法節](https://arxiv.org/html/2510.25257v1#S3)・[公式実装](https://github.com/RT-DETRs/RT-DETRv4) | AIFI 後・CCFF 前の特徴取得、teacher provider、学習可能 projector、蒸留 loss、勾配統計と epoch 境界の制御 |
| D-FINE | box 座標の直接回帰を分布の反復 refinement（FDR）へ拡張し、GO-LSD で layer 間の自己蒸留を行う。[論文](https://arxiv.org/abs/2410.13842)・[公式 decoder](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py) | decoder / refinement state、分布 head・box decode・score 補正、layer ごとの分布出力、対応 criterion と蒸留 |
| DEIM | Dense O2O は mosaic / mixup によって画像中の target を増やし、O2O 枠組みを維持する。MAL を組み合わせて学習する。[論文](https://arxiv.org/html/2412.04234v2)・[公式実装](https://github.com/Intellindust-AI-Lab/DEIM) | augmentation / collate、target metadata、matching policy、classification loss、学習 schedule。matcher の交換だけでは表現しない |
| DEIM + D-FINE | DEIM の公式実装には D-FINE を基盤とする構成がある。[公式構成一覧](https://github.com/Intellindust-AI-Lab/DEIM#1-model-zoo) | D-FINE model variant と DEIM recipe / MAL を構成し、分布回帰 loss・自己蒸留との整合をまとめて検証する |

各手法を単一の `use_v3` / `use_deim` フラグへ落とさず、組合せを factory で表現する。D-FINE 型の回帰表現変更は学習済み RT-DETR に対する等価 transformation pass にはできない。RT-DETRv4 の蒸留方式を別 student に適用することと、公開 RT-DETRv4 のモデル・学習設定を完全再現することも区別する。

### 4.5 注入点と拡張 contract

| 注入点 | 契約と所有者 | 最小の検証 |
|--------|--------------|------------|
| Model components | factory が Backbone / Encoder / QuerySelector / Decoder を受け取り、FeatureSpec と query 入力仕様を照合する | default factory の baseline parity、component 差し替え |
| FeatureTaps | `encoder.after_aifi` など計算上の位置を enum と型で定義し、必要な出力だけ明示的に返す。学習用途は勾配を保持し、log 用 detach と区別する | 取得するだけでは予測が変わらず、補助 loss から対象層へ勾配が届く |
| Query / branch plan | query group の役割、範囲、attention 制約、reference 表現、assignment 方針を宣言する。DN、通常 O2O、追加分岐を別々に識別する | 分岐間の情報漏れ、group ごとの loss 正規化、空 target |
| Decoder / refinement | variant が head と state の型を定める。直接 box logit と分布 logits を区別し、bin 数・支持点・reference box・detach 規約を保持する | 最終 box と中間 state の forward / backward 比較 |
| TrainingMethod | 学習 wrapper が補助 head、projector、DN embedding を登録し、モデルの明示的な stage API を使って分岐を実行する | parameter 登録、DDP、共有 weight の一意性、分岐除去 |
| Assignment / criterion | query group / prediction 型を入力し、対応関係と loss を返す。loss ごとの target、重み、正規化、必要 payload を宣言する | loss 単独比較、不正な組合せの構成時エラー |
| TeacherProvider | teacher の入力変換・特徴位置・出力型・凍結方針を定める。teacher と trainable projector の所有権を分ける | teacher に勾配がなく、student / projector には勾配がある |
| StepController | backward 後の勾配統計取得、optimizer step / epoch 境界での状態更新を明示する | AMP / accumulation / DDP 下の更新回数と resume 一致 |
| ExperimentRecipe | データ変換、target metadata、optimizer groups、schedule、seed、method 構成を解決済み設定として保存する | 同じ設定の再実行、ablation 差分の追跡 |

これらは必要な交換境界を示すものであり、初版から汎用 plugin engine を作る要件ではない。純粋関数、`nn.Module`、小さな Protocol を使い、拡張を composition root に直接渡す。通常の研究追加で monkey patch、module 名の文字列探索、隠れた forward hook に依存しない。hook は一時的な観察用途に限る。

`FeatureTaps` は最終 encoder 出力だけで代替しない。AIFI の出力と CCFF 後の出力には意味の差があるため、論文が参照する位置を明示する。分布回帰では共通境界に最終 normalized cxcywh を返しつつ、内部では variant 固有の state を保持する。既存の狭い contract で表せない場合は、decoder 全体と対応 payload を追加できる。

## 5. Public API と composition

### 5.1 Baseline と deployment

以下は提案 API の概念例であり、現在のリポジトリで実行できるコードではない。

```python
cfg = RTDETRConfig.r50vd(num_classes=80)
model = build_reference_model(cfg)
load_official_checkpoint(model, checkpoint, source=baseline_spec)

model.eval()
with torch.inference_mode():
    output: ModelOutput = model(images, plan=InferencePlan(decoder_layers=6))
    detections = postprocess(output.final, original_sizes_wh)

deployed = compile_model(
    model,
    profile=DeploymentProfile(
        backend="onnxruntime",
        precision="fp32",
        input_shape=(1, 3, 640, 640),
        decoder_layers=6,
        passes=("fuse_conv_bn", "precompute_anchors"),
    ),
)
```

設定は immutable dataclass とし、channels / strides / levels のように導出可能な値は FeatureSpec から計算する。registry を使う場合も、YAML 文字列から任意 object graph を作る用途ではなく、composition root における明示的な factory 選択に限定する。

`compile_model` は元の model を変更せず、検証対象の deployment module と artifact を生成する。profile の layer 数から実行 plan と必要な pass を導出し、同じ設定を複数箇所で指定させない。SDK 未導入、未対応 precision、shape と pass の矛盾は変換開始前にエラーにする。

### 5.2 研究拡張の composition

構成は三つの単位に分ける。

- **ModelVariant:** 推論に残る計算と parameter、内部 state、出力 contract を定義する。RT-DETR、D-FINE、独自 decoder など。
- **TrainingMethod:** DN、補助分岐、蒸留、loss、状態を持つ学習制御を組み立てる。重み共有や分岐の実行順もここで明示する。
- **ExperimentRecipe:** variant / method を選び、backbone、データ拡張、optimizer、schedule、再現対象を固定する。

```python
# 概念例。builders は通常の Python import で外部 package からも渡せる。
recipe = load_recipe("deim_dfine_l_coco")
variant = build_dfine_variant(recipe.model)
method = build_deim_dfine_method(recipe.training)  # 分布 loss と自己蒸留も構成

experiment = build_experiment(
    variant=variant,
    method=method,
    recipe=recipe,
)
validate_composition(experiment)  # 必要な features / payload / group の照合

# 学習後、選択した student / EMA 重みから推論 module を抽出する。
inference_model = experiment.extract_inference_model(weights="ema")
```

Baseline の学習構成には Hydra の階層 config と `instantiate` による明示的な DI を使う。`_target_` で各 component を指定し、optimizer / scheduler は `_partial_` factory として実行時の parameter に接続する。構成処理は application 境界に置き、model 本体は Hydra に依存させない。`build_deim_dfine_method` のような明示的 factory は、必要な分布 loss を維持し、classification loss や augmentation を DEIM の設定へ置き換える。複数 method が同じ loss、query group、controller を所有する場合は構成を拒否し、統合 factory で一つに決める。

各拡張は `ExtensionSpec` に ID / version、提供・要求する contract、parameter / state の所有者、推論に残る module、weight converter、検証済み組合せを宣言する。例えば GO-LSD 用 loss が distribution payload を要求する場合、直接回帰の RT-DETR に付けると構成時に失敗する。能力の一致は組合せの実行可能性を示すだけで、AP の改善や論文再現を保証しない。

baseline の contract / fixture を共通の回帰検証として残し、各 extension は自身の公式実装または定義済み reference と比較する。研究 module の内部 API は試行できる一方、公開する contract の破壊的変更には version と移行例を付ける。

## 6. Training と inference の分離

### 6.1 RT-DETR の標準学習経路

- Core model は train/eval 共通の `ModelOutput` を返す。出力の収集範囲は plan 内の `OutputPolicy` が決める。学習 plan は必要な encoder / decoder / DN 出力を要求し、不足時は criterion 実行前に検出する。
- `model.train()` / `eval()` は BN、Dropout と公式の勾配経路を制御する。実行 layer 数や auxiliary 出力数まで暗黙に変更する用途には使わない。初版では学習時の layer 削減を許可しない。
- DenoisingBuilder、matcher、criterion、loss aggregation は `training/` に置く。core は DN の生成方法や dataset target を知らず、生成済み query と mask を受け取る。
- DN の class embedding は学習 wrapper が所有する `nn.Module` とし、optimizer、DDP、checkpoint に登録する。DN を単なる前処理として切り離し、学習可能 parameter を失う設計にはしない。
- multi-scale resize とその乱数状態は training pipeline に移し、補間方法、適用順序、対象サイズを保存する。初期互換検証では同一の変換済み入力と DN ノイズを両実装へ与える。

学習 wrapper は、DN の構築・連結・予測の分離と criterion への metadata 引き渡しをまとめる。DN 無効または batch 内の全 target が空の場合も明示的に扱う。

resume 用 checkpoint は wrapper 全体の重み、optimizer、scheduler、EMA、AMP scaler、epoch / step、乱数・sampler 状態を保存する。初版では epoch 境界での再開を保証対象とし、任意 minibatch 位置からの完全一致は別要件とする。公式重みを学習に使う場合は、DN embedding を含む training wrapper への変換を行う。

```python
training_model = build_training_model(model, cfg.training)  # DN embedding を所有
training_model.train()
step = training_model(batch.images, batch.targets, plan=TrainingPlan.official())
losses = criterion(step.predictions, batch.targets, step.dn_context)

model.eval()
with torch.inference_mode():
    infer_out = model(images, plan=InferencePlan(decoder_layers=3))
    detections = postprocess(infer_out.final, original_sizes_wh)
```

3-layer 推論は速度・精度の選択肢であり、6-layer 推論との数値一致は要求しない。同じ 3-layer plan を使った reference に対する一致と、6-layer baseline からの AP 差を別々に記録する。

### 6.2 補助分岐・蒸留・学習制御のライフサイクル

TrainingMethod は `features → branch plan → predictions → assignment / losses` の順序を明示し、追加の decoder 呼出しも共有 module に対して行う。v3 型の重み共有は decoder の複製で代替せず、一つの所有者が登録した parameter を再利用する。optimizer と checkpoint で同じ parameter を二重管理しない。

VFM 蒸留では teacher を eval / frozen とし、teacher 用 normalization と student 用前処理の差を TeacherProvider に閉じ込める。mosaic / crop / resize 後に空間的に対応した画像から特徴を得て、どの画像・領域を比較したかを記録する。teacher をキャッシュする場合も元画像 ID だけを key にせず、変換条件と teacher の hash を含める。student feature と projector の勾配は保持する。GO-LSD のような自己蒸留は外部 teacher と同一視せず、各 variant が定めた target layer と detach 規約を使う。

GAM 型の制御を含む StepController には、勾配を観測する時点、AMP scale の補正、accumulation の単位、rank 間集約、clipping との前後関係、係数更新の周期を定義する。標準経路では勾配が揃い AMP unscale を終えた後、clipping / optimizer step の前に観測する。公式実装の再現に別の順序が必要な場合は recipe へ明記して比較する。係数・集計値・epoch 状態は checkpoint に保存し、core forward 内の可変 state にしない。

推論への移行は `extract_inference_model` で行い、teacher、学習専用 projector、補助 head / 分岐、loss、controller を除外する。抽出前の student eval 経路と、同じ重み・plan の抽出後モデルを比較する。比較対象は学習後の student であり、元の RT-DETR checkpoint ではない。D-FINE の分布 refinement など推論に必要な部分は残す。

## 7. Transformation pass 設計

pass は入力 module、前提条件、変換内容、検証区分、backend 制約を持つ。

入力は baseline または検証対象 variant から抽出した推論 module とする。研究上の architecture / objective 変更と deployment の最適化 pass を区別し、pass は対応する variant capability を宣言する。例えば直接 box 回帰を前提とする rewrite を分布回帰 decoder へ自動適用しない。

変換の保証は三種類に分ける。

1. **数値等価:** fusion や固定 shape の事前計算。丸め誤差を考慮した stage parity を要求する。
2. **実行 plan に対する等価:** layer 削減や最終出力だけの export。同じ plan の reference と比較する。
3. **近似:** precision 変更、量子化、近似 kernel。数値誤差と AP 低下の budget を profile ごとに定める。

検証前の artifact は候補として保存できるが、必要な gate を通るまで検証済み profile として配布しない。

| **Pass**           | **変換**                                       | **保証／検証**                       |
|--------------------|------------------------------------------------|--------------------------------------|
| fuse_conv_bn       | Conv と BatchNorm を推論用に融合               | FP32 stage parity                    |
| reparameterize_rep_blocks | RepVGG 系の複数分岐を単一 Conv に統合    | eval 状態、保存済み BN 統計、block / stage parity |
| precompute_anchors | 固定 shape の anchor / valid mask を定数化     | shape profile ごとの完全一致         |
| precompute_position_embeddings | 固定 feature shape の位置埋め込みを定数化 | level / shape / dtype ごとの一致 |
| prune_decoder      | 使用 layer 数と中間出力を削減                  | 指定 layer の出力が reference と一致 |
| select_precision   | FP16 / BF16 / INT8 境界を設定                  | 精度・誤差 budget を記録             |
| replace_self_attention | self-attention の実装を置換                | Q/K/V projection、mask、scale、dropout、数値比較 |
| replace_deformable_attention | sampling 演算を backend kernel / plugin に置換 | 座標、補間、境界処理、layout、数値比較 |
| fold_postprocess   | 必要時のみ backend graph に postprocess を統合 | API contract と end-to-end parity    |

pass manager は適用順序、依存関係、排他条件を検証する。標準順序は plan 固定 → FP32 fusion / 再パラメータ化 → shape 依存定数化 → backend lowering / precision 設定 → export とし、backend ごとの差分は manifest に残す。BN を使う融合は eval と確定した統計を前提とし、事前計算した定数は宣言外の H/W に流用しない。可変 H/W と単一 shape の事前計算を同じ profile で要求した場合は拒否する。

各 pass の比較に加え、全 pass 適用後も同じ plan の reference と比較する。manifest には source commit、config / checkpoint hash、schema version、pass 名・version・引数・順序、shape / precision、toolchain / device、検証結果を記録する。INT8 は後続拡張とし、追加時には calibration dataset と前処理の識別子も必須とする。

## 8. Export / deployment boundary

| **Boundary**       | **用途**                               | **想定 I/O**                               |
|--------------------|----------------------------------------|--------------------------------------------|
| Whole model        | 簡単な配布・検証                       | images → logits, boxes                     |
| Backbone + Encoder + MemoryProjector | 特徴抽出を engine 化         | images → memory, spatial_shapes, level_start_index |
| QuerySelector      | top-k / anchor を custom kernel 等で置換 | memory + spatial_shapes → content, ref_logits, source_indices |
| Decoder            | attention plugin、layer 数最適化       | memory + spatial_shapes + level_start_index + queries → logits, boxes |
| Postprocessor      | サービス API または engine 内処理      | normalized output → image-space detections |

標準 export contract は、重みと定数を保持し、呼出し間に入力依存の状態を更新しない Tensor-only function とする。whole-model の既定出力は `(logits, boxes)`。postprocess 統合時は別 schema とし、`original_sizes_wh` 入力と `(labels, boxes_xyxy, scores)` 出力を明記する。現行 exporter の出力と直接比較する場合は、この統合 schema を使う。

初版は固定 `(B,3,640,640)` を profile ごとに検証する。dynamic batch はその後に追加し、可変 H/W は位置埋め込み・anchor 生成・feature shape 導出を含めて成立した時点で対応を宣言する。`S` は feature shape から導出し、独立した dynamic axis として自由に指定させない。`N`、`Q`、`K`、decoder layer 数は artifact ごとに固定する。

最初の分割は memory 境界とし、前段の Backbone + Encoder + MemoryProjector と、後段の QuerySelector + Decoder を組み合わせて検証する。分割間の転送・同期・Tensor layout 変換を含めて測定し、whole-model より速くなるとは仮定しない。

backend matrix には対応 version、opset、device、precision、shape、必要 plugin、未対応 operator を記載する。未対応 operator を発見したら明示的に失敗させ、CPU fallback を許可する profile は実行場所と転送コストを報告する。

matrix の行は `variant × inference plan × backend profile` とする。共通 `(logits, boxes)` schema を返すことと、内部 operator を export できることは別に検証する。研究 payload と学習専用 branch は標準 export 出力に含めず、未対応 variant の研究実験は eager で継続できる。

## 9. 互換性と再現性

1. **Checkpoint:** source commit、解決済み config、checkpoint の取得元・hash、`model` / `ema` の選択を固定する。key の変換表と shape を検証し、missing / unexpected key は明示的な allowlist 以外をエラーにする。重み変換と optimizer を含む学習 resume の互換性は区別する。

2. **Stage:** backbone、encoder、memory projection、anchor、選択 score / index、query、各 decoder layer の logits / boxes を順に比較する。top-k の同点・僅差では順序差があり得るため、score と選択集合を保存し、差が生じた最初の境界を報告する。順序を合わせて不一致を隠さない。

3. **Training:** 同一 minibatch・DN ノイズで loss と主要 parameter の勾配を比較する。その後、seed、augmentation、optimizer parameter groups、scheduler、EMA、AMP、batch / world size、loss 正規化を揃え、学習曲線と再学習 AP を検証する。

4. **Metric:** COCO val2017 の annotation、category mapping、resize、postprocess、evaluator version を固定する。公式 checkpoint の変換前後の AP と、独立した再学習 run の AP を分けて報告する。公開表の値だけを合否判定の基準にしない。

5. **Export:** 同じ checkpoint・plan・前後処理の eager と backend を比較する。近似 profile は eager baseline からの AP 低下も別途測定する。

| 指標 | 暫定基準と確定方法 |
|------|--------------------|
| FP32 stage parity | component ごとの atol / rtol、最大絶対誤差・相対誤差、非有限値 mask を Phase 0 で固定する。比較失敗後に無断で閾値を緩和しない |
| 変換 checkpoint の AP | 同じ評価条件の公式実装に対し絶対差 0.1 AP point 以内を初期目標とする。AP は 0〜100 表記 |
| 再学習 AP | seed 間のばらつきを測定して許容幅を定める。checkpoint 変換の 0.1 point をそのまま再学習の gate にしない |
| 近似 profile | 許容 AP 低下、数値誤差、目標 latency を profile ごとに事前定義する。未定義の profile は検証済みにしない |
| Reference eager 速度 | 同一環境・同一 plan の公式 eager に対して中央値の低下 3% 以内を初期目標とし、反復測定で揺らぎを確認する |

AP / 速度の数値は未測定の設計目標である。Phase 0 で検証環境と閾値を確定し、変更理由は ADR と比較レポートに残す。

### 9.1 研究比較と再現性の単位

研究用 manifest は baseline の情報に加え、variant / method / recipe の ID・version、拡張 code commit（未コミットなら差分 hash）、source paper / repo、teacher の重み・前処理、追加 pretraining data、query groups、loss 構成、controller 状態、学習時間・GPU hours・peak memory を持つ。学習コストが増えて推論コストが変わらない手法も評価できるようにする。

論文の正式 recipe を再現する比較と、同じ backbone・データ・計算予算で要素を比較する ablation を分ける。例えば `RT-DETR + Dense O2O`、`RT-DETR + MAL`、両者の組合せ、`D-FINE`、`D-FINE + DEIM` をそれぞれ実験 ID として管理する。v3 の分岐、v4 の蒸留・適応係数も個別に切り替えられる recipe を用意し、差分設定を保存する。

seed、実行回数、AP の平均・ばらつき、学習時間、推論条件を併記し、未測定値を成功と扱わない。新規研究 variant に元の RT-DETR との出力一致を要求するのではなく、baseline の非回帰、自身の意味論・勾配、対応する公式実装との一致をそれぞれ検証する。

## 10. テスト戦略

| **レベル**     | **対象**                               | **主な検証**                                  |
|----------------|----------------------------------------|-----------------------------------------------|
| Unit           | box transform、anchor、top-k、mask、DN | shape、値域、境界条件、gradient               |
| Contract       | 全ての module boundary                 | dtype、layout、座標系、dynamic axis           |
| Parity         | 公式実装 vs clean 実装                 | stage-wise numerical comparison               |
| Export         | torch.export / ONNX / backend          | graph build、宣言した shape profile、output parity |
| Training smoke | 固定 minibatch、短い学習、2-rank DDP   | finite loss / gradient、parameter 更新、短期 overfit、resume |
| Reproduction   | 正式 recipe                            | AP、学習曲線、throughput、memory              |
| Performance    | deployment profiles                    | latency p50/p95、throughput、VRAM、build time |
| Extension      | component 注入、学習分岐、特徴蒸留、分布回帰、recipe | baseline 非回帰、payload / capability 整合、gradient、共有 parameter、推論抽出 |

境界ケースには空 target、一部画像のみ target なし、DN 無効、top-k 同点、`S<Q` の拒否、mask 方向、許可外 shape、先頭 / 最終 decoder layer を含める。学習では BN / Dropout の mode、detach 経路、DN embedding の optimizer / DDP 登録、全 rank での loss 正規化を確認する。DDP は出力 schema の一致だけでは保証せず、DN が使われない rank の未使用 parameter も試験する。

テストは実行コストごとに分ける。

- **PR ごと:** unit / contract、小さな FP32 fixture、checkpoint 変換の構造検証。利用可能な固定環境で export smoke を実行する。
- **定期・release 候補:** GPU stage / gradient parity、2-rank DDP、backend ごとの出力比較、COCO 評価、性能計測。
- **recipe 変更・正式 release:** 学習再現と中断 / resume を検証する。短い smoke で正式 AP の再現を代替しない。

golden tensor は少数画像の各 stage 出力を lossless に圧縮し、入力・重み・生成環境の hash と一緒に管理する。大きな fixture は artifact storage に置き、repo には manifest と小さな例を残す。参照実装と公式実装は import 名や依存の衝突を避けて別 process で比較できるようにする。

ベンチマークは model-only と前処理・転送・postprocess を含む end-to-end を分ける。device、driver、runtime、batch、shape、precision、warm-up / 測定回数、同期方法を保存し、latency p50/p95、throughput、peak memory、engine build time を報告する。固定 accelerator 環境で反復測定し、汎用 CI の速度だけで回帰判定しない。

研究拡張の導入時には、①拡張なしの baseline parity、②拡張を有効にした固有 loss / gradient、③学習専用 module を除いた推論 parity、④不正な組合せの拒否を確認する。追加 loss の無効化による非回帰試験では、乱数・BN 更新・追加分岐の副作用も止め、係数をゼロにしただけで等価とはみなさない。

各拡張には独立した検証状態を表示する：`contract-tested`（forward / backward と統合 smoke）、`reproduced`（固定 recipe と metric）、`export-verified`（対象 schema / shape）、`backend-verified`（特定 profile）。contract-tested の段階で研究コードを受け入れ、reproduced と deployment の達成を分ける。小さな実証を論文手法の完全実装として公開しない。

## 11. リポジトリ構成案

```text
pyproject.toml                            # non-package uv workspace and shared tools
packages/
  core/
    pyproject.toml
    src/clean_rtdetr/                     # model, inference, and evaluation
    tests/
  export/
    pyproject.toml
    src/clean_rtdetr_export/              # ONNX adapter, exporter, and CLI
    tests/
  training/
    pyproject.toml
    src/clean_rtdetr_training/            # losses, DN, Lightning, and configs
    tests/
benchmarks/                               # reproducible benchmark harness and manifests
tools/                                    # checkout-only developer diagnostics
experiments/                              # research recipes and ablation definitions
docs/                                     # extension guide, contracts, backend notes, ADRs
```

uv workspace は `clean-rtdetr`、`clean-rtdetr-export`、
`clean-rtdetr-training` の distribution を管理する。export と training は core
にのみ依存し、相互には依存しない。ONNX Runtime と Lightning / SciPy は各
distribution が所有し、core の import・推論環境へ流入させない。workspace
は lockfile を共有するため、依存境界の検証では対象 package を明示して
環境を同期・実行する。repository root 自体は distribution とせず、各 member
が PyPI に独立して build / publish できる metadata、README、source、test を
所有する。public namespace は `clean_rtdetr`、`clean_rtdetr_export`、
`clean_rtdetr_training` とし、旧 subpackage への shim は置かない。

`core/` は contract と geometry など小さな共通処理に限る。各 component の具体的な計算は `models/` に残し、汎用化のための registry や独自 graph framework を追加しない。公式コードを移植する箇所は元の license と attribution を保持する。

`variants/` は推論構成、`methods/` は学習方式、`experiments/` はそれらの組合せと研究固有コードを置く。研究者は外部 package を import して同じ factory 引数へ渡すこともできる。teacher 用 SDK などの optional dependency は利用する method だけが要求し、baseline の import / 推論には要求しない。

拡張ガイドには最小例、必要 contract、source 論文・license、parameter / state の所有者、既知の制約、検証状態、ablation の実行方法を含める。研究用の hook を増設するたびに baseline を変更する運用を避け、繰り返し必要になる境界だけを共通 API へ昇格する。

## 12. 実装フェーズ

| **Phase**          | **成果物**                                    | **完了条件**                        |
|--------------------|-----------------------------------------------|-------------------------------------|
| 0. Baseline       | R50vd の commit / config / checkpoint / 環境 manifest、stage fixture、AP / latency | 比較を再実行でき、各 gate の閾値が確定している |
| 1. Reference core | typed contracts、clean forward、weight converter、明示的 factory | 固定入力の weight / stage / AP parity と component 注入・特徴取得の非回帰が合格 |
| 2. Export 基盤   | Tensor-only schema、固定 shape の whole-model ONNX Runtime | graph 検査、runtime 実行、logits / boxes と AP 比較が合格 |
| 3. Training / 研究基盤 | DN wrapper、criterion、method / recipe composition、DDP / resume | baseline の loss / gradient / AP / resume と研究拡張の最小実証が合格 |
| 4. 分割・変換    | memory 境界の export、pass manager、fusion / 事前計算 / layer 削減 | 同一 plan の parity、結合時の一致、manifest と転送込み benchmark |
| 5. TensorRT      | FP16 profile、必要な adapter / plugin、配布 artifact | profile の AP / latency budget、対応 matrix、再構築手順が揃う |

Phase 1 の途中で最小構成の export 可否も確認し、契約を固める前に未対応演算を把握する。Phase 2 を長期の再学習より先に置くことで、export 上の設計変更を早く取り込む。OpenVINO、Torch-TensorRT、INT8、可変 H/W は初版完了後、用途と検証環境が揃ったものから追加する。

研究拡張は Phase 1〜3 で次の順に構造を実証する。完全再現はそれぞれ独立した milestone として進める。

| 実証 | 最小成果物 | 構造上の完了条件 |
|------|------------|------------------|
| Recipe / loss | Dense O2O 用のデータ経路と classification loss の差し替え例 | model forward を編集せず有効化でき、target / loss の検証が通る |
| 学習分岐 | encoder 補助 head と共有 decoder の追加 query group | parameter の重複登録がなく、branch ごとの mask / gradient が検証できる |
| 特徴蒸留 / 制御 | AIFI feature tap、固定 teacher の小さな代替、projector、状態を持つ係数制御 | student に勾配が届き、teacher / controller の状態と推論抽出を検証できる |
| 回帰表現 | 分布を state に持つ decoder と型付き payload / criterion | baseline の contract を書き換えず、box 出力と分布 loss を学習できる |
| 組合せ | 分布回帰 variant に recipe / classification loss の拡張を適用 | `DEIM + D-FINE` に必要な構成経路があり、未対応 payload は事前に拒否できる |

最小実証の後に、DEIM / D-FINE の個別再現と組合せ、RT-DETRv3 の分岐群、RT-DETRv4 の実 teacher と GAM を順次検証する。実証用 teacher や簡略 decoder の成功を、各論文の AP 再現とは扱わない。

## 13. リスクとトレードオフ

| **リスク**                   | **影響**                     | **抑制策**                                              |
|------------------------------|------------------------------|---------------------------------------------------------|
| 抽象化しすぎる               | 論文実装が再び追いにくくなる | 公開 interface を少数にし、基本 path は直線的に保つ     |
| 型 wrapper が export を阻害  | backend 変換が不安定         | core 型と Tensor-only export schema を分ける            |
| 公式挙動の不具合も継承       | 互換性と修正が衝突する       | 根拠と差分を記録し、修正は別 variant として検証         |
| backend 差分が core に漏れる | 保守性低下                   | pass / adapter 以外から backend package 参照を禁止      |
| 精度再現コストが高い         | 開発サイクル長期化           | small fixture、短期 smoke、正式 reproduction の三段階化 |
| 最適化組合せ爆発             | 検証 matrix が膨張           | プロジェクトが配布する標準 profile の組合せを release gate 化 |
| forward の一致で勾配差を見落とす | 学習 AP が再現しない     | detach、DN embedding、auxiliary loss の gradient parity |
| shape 定数化と dynamic export の衝突 | 誤った anchor / 位置埋め込みを使う | profile の shape 制約を検証し、宣言外入力を拒否 |
| 分割 export の転送負荷       | 局所高速化が全体を遅くする   | whole-model と転送・同期込みで比較する                 |
| 拡張をフラグで積み重ねる     | 分岐と暗黙依存が増大する     | factory による composition、必要 contract と所有権の照合 |
| 共通型を直接回帰に固定する   | D-FINE 型の内部状態を表現できない | variant 固有 payload / decoder を許し、最終検出境界を共有 |
| 学習専用 module が推論へ残る | teacher 依存や追加 latency が混入する | 推論抽出と同じ student 重みでの parity、依存 package の確認 |
| 論文再現と自由な組合せが混ざる | 改善の要因を判別できない | 再現 recipe と ablation を別 ID・検証状態で管理 |

## 14. 初版の完了条件

- [ ] R50vd の主要処理を paper-to-code map から辿れ、query selection、decoder、学習 wrapper の責務が分かれている。
- [ ] 公式 checkpoint の決定的変換、stage / gradient parity、COCO AP 比較を、固定 manifest から再実行できる。
- [ ] 公式 recipe の学習、DDP、中断 / resume が検証され、再学習の AP とばらつきが記録されている。
- [ ] 固定 shape の whole-model と memory 境界の分割 export が動作し、再結合後の出力と AP が gate を通る。
- [ ] decoder layer 数、self-attention / deformable attention の交換点、precision が contract と profile で指定できる。
- [ ] 少なくとも ONNX Runtime FP32 と TensorRT FP16 の検証済み artifact に、profile、pass manifest、数値誤差、AP、性能計測結果が付属する。
- [ ] backend SDK への依存が adapter 内に収まり、追加した backend によって reference の計算意味論が変わらない。
- [ ] 第12節の研究拡張実証が通り、baseline forward や共有学習 loop を編集せずに variant / method / recipe を注入できる。
- [ ] RT-DETRv3 / v4、DEIM、D-FINE の変更点が注入境界に対応づけられ、DEIM + D-FINE を構成する API と未対応組合せの検証がある。
- [ ] 学習専用 branch / teacher を除去した推論モデルの parity と、共有 parameter / controller の保存・再開を検証できる。
- [ ] 外部拡張の最小例、ablation recipe、実験 manifest、拡張ごとの検証状態が用意されている。全派生手法の正式 AP 再現は初版の必須条件に含めない。

## 15. 初期 ADR（Architecture Decision Records）

以下は採用を提案する設計判断である。実装時に各 ADR へ背景、検討した代替案、検証結果を追記する。

| **ID**  | **決定**                                                                     |
|---------|------------------------------------------------------------------------------|
| ADR-001 | Reference、Transformation、Backend の三層構造を採用する。                    |
| ADR-002 | 中間表現は typed Python contract、export は version 付き Tensor-only schema とし、初版では独自 IR を作らない。 |
| ADR-003 | train/eval で出力フィールドを変えず、plan と出力 policy で収集範囲を指定する。BN / Dropout と勾配規約は維持する。 |
| ADR-004 | 高速化は model 内フラグではなく、順序付き transformation pass とする。       |
| ADR-005 | 公式互換と挙動修正を variant として区別し、各 variant に比較対象と検証結果を持たせる。 |
| ADR-006 | layer ごとの prediction head は Decoder が所有し、反復 box 更新と勾配経路を保持する。 |
| ADR-007 | DN builder と学習可能 embedding は training wrapper が所有し、core には Tensor と mask を渡す。 |
| ADR-008 | 等価変換、実行 plan の変更、近似変換で検証基準を分ける。元の reference model は変換で変更しない。 |
| ADR-009 | 固定 shape の R50vd と ONNX Runtime を先行し、TensorRT 以外の追加 backend と可変 H/W は後続とする。 |
| ADR-010 | 研究拡張は ModelVariant × TrainingMethod × ExperimentRecipe で構成し、constructor / factory から注入する。 |
| ADR-011 | 最終 DetectionOutput を共有し、回帰 state・研究 payload・勾配規約は variant ごとに version 管理する。 |
| ADR-012 | 特徴取得、query group、学習分岐、teacher、学習制御を明示的 contract とし、parameter と可変 state の所有者を一つにする。 |
| ADR-013 | architecture / objective を変える研究拡張と、学習後モデルを変換する deployment pass を分ける。 |
| ADR-014 | 拡張の研究検証、論文再現、export、backend 検証は独立した状態として公開する。 |

Phase 0 で確定する項目は、checkpoint の取得元と hash、`model` / `ema` の選択、toolchain / GPU、stage ごとの許容誤差、学習再現の試行数、TensorRT profile の AP / latency budget とする。未確定値は manifest 上でも未確定と表示し、実測値と混在させない。
