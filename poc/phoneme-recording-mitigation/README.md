# 音量前処理と発話長の制御比較

録音品質診断に続き、同じencoder・登録区間・照合音声を使って次を分けて測る。観測済みのvalidation/testを開発診断として使用し、新規独立評価とはしない。

- 音量前処理: 発話全体のactive RMSを−25dBFSへ揃える単一gain。activeは前回と同じ20msフレーム、max(P90−25dB,−50dBFS)より大きい区間。gain上限±24dB、peak絶対値0.98以内。無音は増幅しない。登録・照合の両方を処理し、PCM16へ量子化する。元の採用区間と境界は保持し、再alignment/QCは行わない。既存モデルと前処理の相性を測る診断であり、end-to-end改善の保証ではない。
- JVS: 前回と同じvalidation/test各150発話、clean/−12dB/雑音20dB/雑音10dB/帯域/残響の6条件。clean validationで元方式/正規化方式の閾値を別々に校正し、加工条件へ固定適用する。条件別の閾値再校正はしない。
- Common Voice: 前回の校正30話者・test30話者、全900照合発話/集合と登録音声。方式別CV校正とJVS閾値の移植を併記する。
- 発話長: 同一JVS原録音の最初の採用音素raw startから1秒・2秒・3秒のnested window。3秒確保できる共通集合だけを元音声との比較に使う。windowへraw境界全体が入る音素だけ保持し、途中音素を切断してラベルを付けない。既知の元alignmentを使った診断で、短音声を実際に再alignmentした結果ではない。元音声全150発話で決めた閾値を固定し、短音声用には調整しない。

各比較は5母音必須と4母音以上。入力不足を拒否として残したFAR/FRR、coverage、EER、採点不能件数を報告する。発話長比較では短い窓でも採点できる同一query集合で元音声を再評価し、支持不足と採点後の差を分ける。対応付き話者bootstrap 2,000回。多数の探索的比較を採用判断には直結させない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-mitigation/experiment.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-mitigation/experiment.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-mitigation/experiment.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-mitigation/summarize.py
```

前回のquality検証環境（SciPy 1.16.3を含む）を使う。元音声とモデルは上書きしない。登録音声の原alignmentを保持するため、正規化後の完全な登録/照合パイプライン検証は別に必要。
