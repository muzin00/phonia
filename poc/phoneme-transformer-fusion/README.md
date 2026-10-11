# 固定encoderでのTransformer音素統合

[Issue #22](https://github.com/muzin00/phonia/issues/22) の録音ごとの音素寄与を実測する。
既存の全36音素encoderと登録上限30区間/音素を固定し、等重み・音素別MLP・音素間Transformerを比較する。
encoderの再学習や初期PoCの条件変更は行わない。

- [欠損・ノイズを含むブラウザ用の結果表](summary.html)
- [結果の解釈](interpretation.md)
- [全seedと信頼区間の結果表](evaluation-results.html)
- [測定結果と条件](evaluation-results.md)
- [機械可読の結果](evaluation-results.json)
- [事前固定した設定](config/protocol.json)

MLPは音素ごとの520次元入力を64次元に変換し、64次元の隠れ層から重みを生成する（追加40,003パラメータ）。
Transformerも同じ入力を64次元に変換し、4 head・2層・FFN 128次元から重みを生成する（追加102,787パラメータ）。
位置encodingは使わず音素ラベルembeddingを加える。欠損音素はattentionとsoftmaxでマスクする。
両方式は元の同音素cosineの加重平均を返す。正の温度とbiasは学習のBCE用で、公開照合スコアには加えない。

入力は登録/照合の128次元embedding、絶対差、積、双方の4統計（log区間数、log平均秒数、1−平均embedding norm、平均RMS/50）である。
照合側の平均embeddingは再正規化せず、登録側だけを正規化する元のcosine計算を維持する。
モデルは話者ID、データセットID、正解ラベルを入力として受け取らない。

JVS70話者＋Common Voice70話者を学習に用いる。各話者35種類のWAVを登録側、残りを照合側に固定する。
JVSはparallel100を登録側で優先し、その中の順序およびCommon VoiceはSHAによる固定順序を使う。
同一WAV SHAの重複は片側にまとめ、validation/testの話者・音声SHAとの交差を禁止する。
登録は音素ごとに実在する区間をsource-diverseで最大30個選ぶ。音素不足は複製して埋めない。

1更新は32照合発話、それぞれ本人1・同corpusの別話者1の64ペア。本人/他人の欠損マスクをそろえる。
seedごとに4,000更新分のペア列を事前保存し、MLPとTransformerへ同じ順序で渡す。
学習seedは3個。AdamW、cosine learning rate、BCE＋一様重みへの小さいKL正則化を使う。
全モデルを4,000更新まで学習し、500更新ごとのvalidation通常EER最小で採用checkpointを決める。
同点は早いcheckpointを選び、初期の等重みcheckpointも候補に含める。
validation通常発話のFAR 1%、FAR 0.1%、EER operating閾値を固定する。
6モデルと閾値のSHAを一括凍結してからtestを採点し、全seedを公開する。

```bash
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-transformer-fusion/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-transformer-fusion
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-transformer-fusion

# 既存のrunを上書きしない。全コマンドへ同じ未使用の--runを渡す。
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/study.py prepare --run artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/study.py train --run artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/study.py evaluate --run artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/verification.py --run artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/report.py --run artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1

# この固定runの追加診断。未使用のstressサブディレクトリが必要。
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/stress.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/diagnostics_control.py
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-transformer-fusion/summary.py
```

`artifacts/`はGit管理外。新規run内に入力集約、ペア列、設計freeze、学習履歴、採用checkpoint、閾値、全照合行、bootstrap、独立監査を保存する。
元の音声・alignment・特徴とencoderのSHAを準備時および学習後に検証する。
公開レポートにもSHAを付ける。独立監査は元のembeddingから音素別cosineを再計算し、重み付きスコア、EER、FAR/FRR、共有話者bootstrapを確認する。

test平均EERは等重みの通常1.497%・別文1.020%に対し、Transformerは1.186%・0.941%。
MLPは3 seedとも学習後のvalidation EERが改善せず、初期の等重みを採用した。
Transformerは子音50%欠損の別文では改善したが、20%の照合区間へ10dBの白色ノイズを加えると3.356%・3.401%へ悪化した。
追加診断は同じ6モデルとclean validation閾値を使い、追加学習・閾値調整・testによるseed選び直しは行わない。
ノイズなしの51区間を元WAVから再計算したembeddingはcacheと完全一致し、登録側と未選択の照合区間はすべて不変である。
5母音不足による無採点は全条件で共通。合成ノイズ診断では既存の境界を保持し、再alignmentは行わない。

重みは照合への寄与であり、録音品質や生体依存度の確率ではない。
参考: [Attentive Statistics Pooling](https://arxiv.org/abs/1803.10963)、
[Attention Back-end for Multiple Enrollment Utterances](https://arxiv.org/abs/2104.01541)。
これらは異なる入力粒度・評価条件の先行研究であり、今回の改善を保証するものではない。
