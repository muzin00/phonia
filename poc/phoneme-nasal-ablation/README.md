# /m/・/n/ 個別追加と併用の切り分け

同じ[7音素学習済みモデル](../phoneme-consonant-training/README.md)を使って、
5母音、5母音＋m、5母音＋n、5母音＋m/nを比較する。新規学習は行わない。
前回の[使用時間を揃えた比較](../phoneme-consonant-evaluation/README.md)を固定したまま、
単独追加の2条件を追加する。

## 事前固定した設計

| 項目 | 条件 |
| --- | --- |
| モデル | JVS＋Common Voice、140話者ラベル、7音素、30,000更新、seed 20260926 |
| 発話集合 | 前回common7をそのまま共有。validation通常525／別テキスト279、test通常525／別テキスト265 |
| 評価話者 | validation15／test15、JVS train70と分離 |
| 登録候補 | 同じ母音各10区間と、その候補元WAV内のm/n各10区間。抽出seed 20261010 |
| 使用時間 | 登録最大72,000frame、query最大24,000frame、各24kHz。元の母音候補が少ない場合は同じ短い予算 |
| 区間処理 | 前回と同じ30〜250ms、音素時間を5msずつ均等割当、境界内center crop |
| 子音QC | 30ms以上・RMS−50dBFS以上。N/my/nyをm/nへまとめない |
| スコア統合 | 音素内で区間cosineを平均し、音素間で等重み平均 |
| 閾値 | 各条件のvalidation通常文のみでFAR1%・0.1%・EER動作点を固定 |
| 主指標 | testの固定閾値FAR/FRR（validation目標FAR1%）と診断EER、通常文／別テキスト別 |
| 対応付き差 | 全6組を固定。15話者、10,000回、前回と同じbootstrap draw |
| 基準の再現 | 5母音とm/n併用の区間を完全一致させ、再推論の絶対誤差≤1e-6を検査後に元スコアを再利用 |

単独条件が予算に収まらない場合は停止し、対象発話を黙って減らさない。
同じ6音素であるm単独対n単独が音素の種類の切り分け、併用対各単独が追加効果の比較になる。
母音・子音への時間配分、区間数、実際のPCM、融合重みの変化を含むので、
純粋な音素種類数の因果効果は特定しない。他の子音組は次の検証対象である。

今回のデータ・音素・モデルは採用済みのものを再利用する。新しいデータセットや音素を
学習へ追加するときは[データ採用手順](../../docs/training-data-adoption.md)に従う。

## 再現

設定は[protocol.json](config/protocol.json)。元の凍結ファイルを変更せず、
新しい生成先 `artifacts/phoneme-nasal-ablation/matched-20261010-v2/` に保存する。
prepareは空の生成先のみ許可し、設計・入力・元データ・モデル・コードを新規推論前にSHA256で凍結する。
validationの閾値・スコアを凍結するまで新しいtest推論は開始できない。

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-nasal-ablation/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-nasal-ablation/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-nasal-ablation/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-nasal-ablation/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-nasal-ablation/report.py
```

CIは多重比較補正をしていない探索的な区間。既に観測済みのJVS testと1学習seedを使用する。
独立holdoutや未知の録音条件における確証とは扱わない。

## 完了した実測

2026-10-10に全4条件のvalidation・test、全6組の対応付き差の測定を完了した。
通常文のEERは5母音2.395%→m単独1.728%／n単独2.027%／併用1.333%、
別テキストでは1.509%→1.186%／1.456%／0.889%だった。
併用が良好な観測値を示す一方、m対nの主指標差は全て95% CIが0を含む。
厳しいFAR動作点ではFARとFRRの交換もあるので、全面的な優位とは解釈しない。

[結果の解釈](interpretation.md)・[ブラウザ用比較表](evaluation-results.html)・
[全結果と対応付きCI](evaluation-results.md)・[全48セルCSV](evaluation-cells.csv)・
[全指標JSON](evaluation-results.json)を保存した。

7件の単体テストとRuffを通過した。artifacts内の`independent-audit.json`では、
48セルの件数と率、16セルのEER、10,400件のbootstrap直接再計算、
168組の差配列とCI、47,820件の前回基準スコア再利用を独立に照合した。
HTMLの表構造とリンク先の存在を確認した。ブラウザでの実表示は未確認。
初回v1の保存失敗を保持し、条件を変えず新規凍結したv2で完了した。
