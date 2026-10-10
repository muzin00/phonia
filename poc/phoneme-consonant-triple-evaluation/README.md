# m+n+sを追加した同じ音声時間での比較

[前回の4条件](../phoneme-consonant-combination-evaluation/README.md)に、母音5種＋m/n/sを追加する。
既存の8音素モデルを固定し、追加学習は行わない。

| 条件 | 使用音素 | 種類数 |
| --- | --- | ---: |
| vowels5 | a i u e o | 5 |
| vowels_mn | a i u e o m n | 7 |
| vowels_ms | a i u e o m s | 7 |
| vowels_ns | a i u e o n s | 7 |
| vowels_mns | a i u e o m n s | 8 |

## 比較条件

- 同じJVS・Common Voice各70話者、30,000更新、1 seedで学習済みのencoderと特徴量統計を使用する。
- 前回のcommon8発話、話者分割、登録候補、4条件の選択PCM区間が完全一致することを推論前に確認する。
- 登録最大3秒・照合最大1秒。各登録・各照合の実使用PCMフレーム数を5条件で完全一致させる。
- 母音・m/nの処理と、/s/のcenter crop後のRMS判定は前回と同じ。無音pad・区間重複・文脈追加は行わない。
- 区間cosineの音素内平均、音素間の等重み平均を使用する。
- validation通常文で条件別にFAR 1%・0.1%・EER動作点の閾値を固定し、testと別テキストには再校正せず適用する。
- 前回のスコアを再利用せず5条件すべて再推論し、既存4条件の再現誤差を検査する。
- 同じ15話者・10,000 bootstrap drawで全10組を比較する。主指標はFAR 1%動作点のFAR/FRRとEER。

前回と同じvalidation通常文495件・別テキスト217件、test通常文494件・別テキスト220件を対象とする。
全8音素が使える発話に限った比較であり、音素不足を含む全発話での性能は対象外。
総音声時間は同じだが、音素ごとの時間・区間数・選択PCM・統合比率が変わるため、種類数だけの因果効果は確定しない。
観測済みtest・1 seed・15評価話者、多重比較補正なしの探索的CIである。

## 実行

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-consonant-triple-evaluation/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/study.py inputs
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/report.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-triple-evaluation/audit.py
```

設定は[protocol.json](config/protocol.json)、出力先は`artifacts/phoneme-consonant-triple-evaluation/matched-20261010-v2/`。
使用済み出力先への入力再抽出・design-freezeの再作成は拒否する。

2026-10-10に全5条件のvalidation・test・全10組の比較を完了した。
既存4条件の全85,560試行を再推論し、スコア・音素別スコアの最大絶対誤差は1.541×10⁻⁷以下。
FAR/FRR・誤り件数・EERは前回と一致した。7テストとRuffチェックが成功した。
`audit.py`は106,950試行から60組の率・20組のEER・10個のvalidation FAR閾値を独立に再計算し、
280組のbootstrap差分配列と区間、全60セルCSV、入力planとfreezeを検査する。

v1の実測は完了したが、監査で前回閾値の微小な丸め差まで完全一致を要求していた。
誤り件数・率・EERの完全一致を保ち、閾値の比較だけ1×10⁻⁶の許容差を設けてv2を再実行した。
v1の入力・実測・監査失敗理由・公開前のレポートはartifacts内に保存している。

[比較HTML](evaluation-results.html)・[結果の解釈](interpretation.md)・[全指標](evaluation-results.md)
