# 使用音声量を揃えた5母音と /m/・/n/ 追加の比較

[学習済み7音素共通encoder](../phoneme-consonant-training/README.md)で、入力音素を追加する効果を実測する。
学習条件・重み・前処理は共有する。新しい学習や、testに合わせた条件変更は行わない。

## 固定条件

| 項目 | 条件 |
| --- | --- |
| 方式 | 5母音のみ／5母音＋取得できたm/n |
| encoder | JVS＋CV、140話者ラベル、7音素、30,000更新、seed 20260926 |
| 評価話者 | JVS validation15／test15。train70と分離 |
| 入力 | 既存の各splitの通常文750＋別テキスト450発話 |
| 登録予算 | 各話者で最大3秒、元の5母音候補の総時間との小さい方 |
| 照合予算 | 各発話で最大1秒、元の5母音候補の総時間との小さい方 |
| 時間一致 | 同じ話者・同じ発話で両方式の実使用PCMフレーム数を完全一致 |
| 登録元WAV | 既存の5母音各10区間の候補元WAV集合を共有。m/nもそのWAV内から選ぶ |
| 子音の登録候補 | 各10区間、元WAVを分散する固定seed抽出 |
| 区間処理 | 30〜250ms。対象区間内center crop。phone時間を5ms単位で均等に割り当てる |
| 子音のQC | 学習と同じ30ms以上・RMS−50dBFS以上。N/my/nyをm/nへまとめない |
| 登録profile | 音素ごとにembeddingを平均してL2正規化 |
| query score | 区間cosineを音素内平均、使用した音素間で等重み平均 |
| 閾値 | 条件・対象集合ごとのvalidation通常文からFAR1%／0.1%／EER動作点を校正して固定 |
| CI | 15話者、10,000回の同じbootstrap drawで方式差を計算 |

使用時間は、実際にencoderへ入れた区間のPCMサンプル数で数える。
同じ区間を繰り返したり、無音を追加したりして時間を合わせない。
音素ごとの時間と区間数、選ばれるPCM内容は方式で異なる。

## 対象集合と音素不足

主比較の `common7` は追加子音側で全7音素が使える同じ発話に限定する。
両方式の発話・claimed話者集合と登録・照合の使用時間を一致させる。
各方式の閾値も同じ対象集合のvalidation通常文から決める。

補助の `native` は全発話を含め、追加子音側は取得できたm/nを使用する。
m/nがない場合や、同じ時間予算内に追加音素の最低30msを収められない場合は5母音で照合する。
5母音不足の場合は両方式ともno_scoreとなり、全入力FRRでは拒否として扱う。
追加子音側の登録profileは常に7音素へ同じ総予算を配分するため、母音profileの時間は5母音方式より少なくなり得る。

この2つの対象集合を混ぜて優劣を判断しない。
`common7`は全7音素がある入力での識別力、`native`は音素不足と可変音素数を含む実用上の挙動を示す。

## 実行と監査

```sh
MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-consonant-evaluation/tests -v
MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-evaluation/study.py prepare
MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-evaluation/study.py validation
MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-evaluation/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-evaluation/report.py
```

設定は[protocol.json](config/protocol.json)。生成先は
`artifacts/phoneme-consonant-evaluation/matched-20261010-v1/`。
全候補と選択済み区間、使用時間、音素カバレッジ、元WAV・アライメント・コードのchecksumを凍結する。
validationの閾値とscoreを凍結しない限りtest推論はできない。
3回の同じbatch推論の完全一致、重み不変、元の音素区間内のcenter crop、重複・重なりの不存在を検査する。
全試行の話者・genuineラベル・音素score・平均score・使用時間も確認する。
reportも凍結時と同じtorchのCPUスレッド数1で実行するため、`OMP_NUM_THREADS=1`を指定する。

## 完了した実測

2026-10-10にvalidation閾値の固定、test推論、10,000回の対応付き話者bootstrapまで完了した。
主比較の通常文525件では、5母音→m/n追加でFAR 1.578→1.007%、FRR 4.571→2.095%、EER 2.395→1.333%だった。
別テキスト265件でも3指標の観測値は下がった。主比較では通常文FRRの差の95% CIのみが0を含まず、
FAR・EERと別テキストの差は不確実性を残す。

[ブラウザ用比較表](evaluation-results.html)、[測定表と監査](evaluation-results.md)、
[結果の解釈](interpretation.md)、[全48セルCSV](evaluation-cells.csv)、[全指標JSON](evaluation-results.json)を参照する。
生score、対応付きbootstrap配列、閾値・入力凍結、独立した件数再集計は上記artifactsに保存した。
`independent-audit.json`では48条件の率・件数、56組の差・CI、CSVとchecksumを照合した。

同じ7音素学習済みモデル内での比較であり、5母音だけを学習したモデルとの比較ではない。
音素の追加と時間配分・区間数・等重み統合の変化を含み、音素の種類だけの因果効果は分離しない。
既に観測済みのJVS testでの探索的な追加実測で、独立holdoutや未知録音条件での確認は別途必要。
