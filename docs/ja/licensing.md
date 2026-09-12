# ライセンスの適用範囲と出典

[English version](../licensing.md)

本リポジトリの Apache-2.0 は、Clean RT-DETR の package・recipe のソースコードと、
帰属表示に記載した Apache ライセンスの改変部分に適用する。外部依存、取得した
重み、データセット、別途出典を示す画像のライセンスを置き換えるものではない。

| 対象 | 適用するライセンス・条件 |
| --- | --- |
| core・export・training のソース | Apache-2.0。package の LICENSE と同梱の出典表示を保持 |
| DEIM・D-FINE・RT-DETRv3・RT-DETRv4 の recipe ソース | Apache-2.0。各 LICENSE、NOTICE.md、ファイル内の変更表示を保持 |
| DINOv3 のコード・学習済み重み | Meta が別途提供する DINOv3 License |
| Penn-Fudan の画像・注釈 | データ提供元の条件。配布ページに Apache-2.0 の許諾は確認できない |
| サンプル写真と検出結果画像 | 元写真は CC0。[画像の出典](../../assets/samples/README.md)を参照 |
| その他の依存ライブラリ・取得した checkpoint | それぞれのライセンスと配布物固有の条件 |

## 参照ソースのライセンス

次の固定 revision で確認している。ソースリポジトリのライセンスを、その
リポジトリで紹介される全重み・全データにも一括して適用しない。

| 参照元 | 固定版のソースライセンス |
| --- | --- |
| RT-DETR | [068dfde6](https://github.com/lyuwenyu/RT-DETR/blob/068dfde65f2667ad6555883c69d73de886518cad/LICENSE)、[29320b6f](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/LICENSE)：Apache-2.0 |
| DEIM | [09d35d53](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/LICENSE)：Apache-2.0 |
| D-FINE | [956d1709](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/LICENSE)：Apache-2.0 |
| RT-DETRv3 | [349e7d99](https://github.com/clxia12/RT-DETRv3/blob/349e7d99a5065e7b684118912e6a74178d4f4625/LICENSE)：Apache-2.0 |
| RT-DETRv4 | [55fefaae](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/LICENSE)：Apache-2.0 |

[Apache-2.0 第4条](https://www.apache.org/licenses/LICENSE-2.0)に従い、再配布では
ライセンス本文、該当する元の著作権・帰属表示、変更したファイルの明確な変更表示を
保持する。上流に NOTICE がある場合は、利用部分に関係する帰属表示も保持する。
自分の著作権表示を追加しても、元の表示を置き換えることにはならない。

core と training の配布物には `licenses/` 以下に出典表示を含め、checkout の
`third_party/RT-DETR` に参照ソースを保持する。recipe を独立してコピーするときは、
ソースや参照 test fixture と一緒に LICENSE と NOTICE.md もコピーする。

## DINOv3 と学習成果物

[DINOv3 License の2025年8月19日版](https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md)
は DINOv3 のコードと重みに適用される。第1条(b)(i)は DINO 素材とその派生物を
再配布する際に同契約の適用と契約書の添付を要求し、第1条(b)(ii)は研究発表での
利用の明示を要求する。実際の checkout・重みに付属する条件を確認する。
recipe の Apache ライセンスがこれを置き換えることはない。

RT-DETRv4 recipe は、利用者が取得したローカルの teacher を読み込む。
本リポジトリに DINOv3 のコード・重みを登録していなくても、training wrapper は
teacher を submodule として登録する。そのため、学習モデル全体の state dict と
Lightning の再開用 checkpoint には teacher の重みが含まれる。EMA を有効にした
場合はその保存状態にも含まれる。DINOv3 を使った checkpoint 全体を
Apache-2.0 のみの配布物として扱わない。

student detector だけを抽出すれば、そのファイルへの teacher tensor の同梱は
避けられる。ただし、それだけで蒸留モデルのライセンスが確定するわけでは
ない。teacher の revision、重みの出典、適用条件を実験記録に残し、公開する
成果物に応じて確認する。`SmokeTeacher` はランダムなローカルテスト用
パラメータを持ち、DINOv3 素材を含まない。学習済み DINOv3 の検証にはならない。

## データセットと変換済み重み

[Penn-Fudan の配布ページ](https://www.cis.upenn.edu/~jshi/ped_html/)にはデータと
出典が掲載されているが、Apache-2.0 の明示的な許諾は確認できない。
loader のコードライセンスは、画像・注釈の利用権を与えない。データは別途取得し、
想定する利用・再配布の条件が不明な場合は提供元へ確認する。

checkpoint の tensor 名の変換や graph の export によって、元の重みの権利関係は
変わらない。処理に使う Clean RT-DETR のコードとは分けて、重みの提供元、
配布物、ライセンスを記録する。
