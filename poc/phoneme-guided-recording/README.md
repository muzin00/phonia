# 母音を確保する指定文と追加録音の検証

短い録音での採点不能を、受付母音数の緩和やモデル変更なしに減らせるか調べる。元の全36音素encoder、等重み、登録上限30区間、5母音必須を固定する。既存録音と元の音素抽出結果によるオフライン再生で、ユーザーへ案内して新たに録音した実験ではない。

## 指定文の比較

JVSの既存照合文50種類（VOICEACTRESS100_051〜100）のうち、train話者の録音時間中央値が6秒以内、G2P上の各母音が2回以上ある文から、時間中央値の短い5文を選ぶ。文と時間の選択はtrainだけを使い、評価の区間数や照合スコアで選ばない。残り45文から、5文全体のtrain時間中央値差を最小にする1対1割り当てで対照を選ぶ。対照に母音数の制限は置かない。

validation/testの各15話者について各5文を全数評価する。校正は元の通常文50文×15話者を使い、指定文と対照へ同じ閾値を固定適用する。実際の評価録音時間・採点率・母音区間数・FAR/FRRを比較し、時間の一致はtrain上の近似マッチングであることを明示する。文章内容も異なるため純粋な母音数だけの因果効果ではない。学習済みの文・既観測話者での候補比較であり、任意の新しい指定文へ一般化しない。

## 追加録音の比較

既観測のCommon Voice 2集合（CV-A/CV-B、各校正30/test30話者）を別々に評価する。各話者の照合30発話を固定hash順へ並べ、各発話を開始点とする最大3発話の循環列を作る。900開始点すべてを残す。元録音が複数episodeへ現れるため、独立900人分とは扱わず、対応付きbootstrapは話者単位とする。追加候補の選択はスコア、発話内容、対象話者との似かよりを参照しない。

- `single`: 最初の1発話。
- `fixed2` / `fixed3`: 常に2/3発話を結合。
- `adaptive1`: 5母音がそれぞれ1区間以上得られた最初の発話境界で停止。最大3発話。
- `adaptive2`: 5母音がそれぞれ2区間以上得られた最初の発話境界で停止。最大3発話。

結合は各音素の実区間embeddingをまとめて平均し、音素間は従来どおり等重み。個別録音ごとの合否をOR結合せず、episodeに対して1回だけ照合する。停止判断に使うのは採用済み区間数のみで、照合スコアやclaimed speakerは使わない。3発話で目標が満たせなくても入力を除外しない。全armで受付は5母音各1区間以上のままなので、adaptive2の目標未達でも5母音が揃っていれば採点する。登録音声は追加候補に含めない。

同じ最大3発話を対照に置き、実消費時間と録音回数を必ず報告する。適応停止と固定回数では実際の平均時間が異なるため、FAR/FRRだけで優劣を判断しない。複数の発話内容・時間・区間数が同時に増える実験であり、母音の存在だけの効果とは主張しない。ユーザー待ち時間や案内画面の操作時間は含まない。全発話の話者が同一というオフライン仮定で、話者交代や音声再生による攻撃への安全性は評価しない。

## 事前基準と校正

各CV集合・各armのvalidationだけで、採点可能な他人試行に対するFAR0.1%と1%の閾値を定め、全閾値を固定してからtestを採点する。主評価はadaptive1対single、FAR0.1%校正。1%校正およびadaptive2/固定回数は補助診断で、主評価不合格を覆す候補選びには使わない。

主評価では両CV集合とも、対応付き話者bootstrap2,000回によるFRR差の95%上端<0、FAR増分の上端≦0.1pp、平均録音時間がsingleの2倍以内、95パーセンタイル≦20秒を満たす場合に限り、実録音のプロトタイプ検証へ進める根拠が得られたとする。これは研究用のスクリーニングで実運用の許容リスクではない。未採点を本人拒否へ含め、除外による見かけの改善を防ぐ。

元PCM特徴・encoder出力・スコア・停止位置・入力非重複・閾値を再計算する。既観測の2集合であるため、片方を新規holdoutと呼ばない。元音声・旧結果・モデルは上書きせず、新しいrunへ固定設計と結果を保存する。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-guided-recording/tests
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-guided-recording/experiment.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-guided-recording/experiment.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-guided-recording/experiment.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-guided-recording/summarize.py
```
