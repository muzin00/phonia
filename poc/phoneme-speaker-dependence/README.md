# 固定14音素モデルの話者依存性を群分けする

追加学習せず、既存14音素encoderが取り出す安定した話者情報を分散成分で評価する。
解剖学的形状と発音の癖を分離した「生体依存度」の直接測定ではない。
同じモデル・同じ15話者・同数の音素区間で、validationから群を決め、別15話者のtestで傾向を検証する。

## 固定条件

- モデル：JVS70＋Common Voice日本語70、30,000更新、65,920 parameters、128次元、14音素。
- 音素：a i u e o m n t d k g s z h。元の重み・特徴統計を固定し、追加学習・モデル選択なし。
- 分散推定：通常文の各話者×音素12区間。区間は別々の発話から選び、元境界内30〜250ms。
- 特徴：L2正規化した128次元embedding。短い区間への一律cropを分散推定には使わない。
- 共変量：log区間長、前後の母音別/調音種別。切片と共変量の係数は音素ごとに推定。
- 分散：多変量・等方的な話者random interceptモデルのREML。128次元共通のB/W比を仮定する。
- スコア：R=B/(B+W)。B/Wは話者間/内のtrace分散成分で、話者平均の標本分散をそのままBにしない。
- 群分け：validation Rだけで、小さい順に並べた全境界の二群内平方和を評価。各群最低2音素。
- 所属安定性：validation話者の1,000 bootstrap。RのCIと高群への所属割合を表示する。
- 評価：各音素で登録5区間×50ms、通常文照合10区間/話者、別テキスト3区間/話者。
- 評価crop：t/d/k/gは元境界の終端、他は中央。RMS−50dBFS以上、pad/増幅/文脈追加なし。
- 発話抽出：固定seed、queryは音素ごとに異なる元WAVから1区間ずつ。音素間の発話は同一ではない。
- 閾値：音素別validation通常文のみでFAR1%/0.1%とEER動作点を固定し、test前にchecksumで凍結。
- 主検証：test通常文の単音素EERを音素等重みで群内平均し、高群−低群の対応付き話者bootstrap95% CI。
- 補助：別テキスト、FAR/FRR、test Rの再現性、validation R対test EERのSpearman相関。
- test bootstrap：2,000回。本人試行はn_i、他人試行はn_i*n_jで重み付け。CIは固定validation群に条件付く。

区間数と実時間は単音素評価で全14音素・全15話者に揃える。分散推定の自然区間長は異なり、共変量で調整する。
前後の具体的な単語、話速、発声状態は完全には統制できない。録音session情報がなく、話者に固定した録音差や発音習慣もBに混ざり得る。
二群クラスタリングは常に二群を返すため、自然に二峰へ分かれているという証拠ではない。
群の音素数は同数ではなく、主結果は単音素EERの群内平均。群を融合した照合性能や母音への追加効果を測った結果ではない。
testは過去にも観測済みのJVS、1モデル/1 seedであり、独立holdoutの確証ではない。

事前の入力のみの確認で、全14音素が揃う50ms発話はvalidation28/test35件、一部話者0件だった。
30msでは全話者を含むが67/69件に限定される。今回は同じ話者・音素別に均等数を抽出する。
通常文50ms区間の最少供給はvalidation14音素中15発話/test11発話、別テキストは各3発話。
登録の全話者最少供給はvalidation7/test8区間であり、登録5・通常文10・別テキスト3をスコアを見る前に固定した。

## 再現

設定は[protocol.json](config/protocol.json)。元のPOC・既存artifactを変更しない。

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-speaker-dependence/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/audit.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/report.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-speaker-dependence/verify_publication.py
```

`prepare`で元PCM、raw境界、学習音声のSHA-256と話者の分離、コード・入力・モデルを凍結する。
v1はモデルの音素一覧をtupleとlistで比較して不一致と判定したため、forward推論前に終了した。
旧ソースと設定を`abandoned-before-inference/`へ保存し、形式を揃える修正後に同じ入力条件のv2を固定した。
validationは群・閾値・bootstrapをtest推論前に凍結する。使用済みrunへの再実行は拒否する。
推論は各batchを反復してbitwise一致と重み不変を検査する。
結果の集計前に独立監査を実行する。

方法の参考：[ICCのモデル選択](https://pubmed.ncbi.nlm.nih.gov/18839484/)、
[話者内・話者間の声の変動](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2019/papers/ICPhS_1509.pdf)。

## 実測完了

高群の通常文macro EERは12.373%、低群は18.149%。高−低の95% CIは[−7.462, −3.698] pp。
別テキストも高群10.873%、低群17.917%と同方向。単音素平均の比較であり、群を融合した性能ではない。
[結果の解釈](interpretation.md)・[HTML比較表](evaluation-results.html)・[CSV](evaluation-cells.csv)・[JSON](evaluation-results.json)。
