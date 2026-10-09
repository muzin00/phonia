# SRC4VCでの追加学習結果

| 学習条件 | 学習話者ラベル | 採用update | 採用基準 | JVS validation EER |
|---|---|---|---|---|
| JVS70 現行モデル（参考） | 70 | 15,000 | 既存のvalidation best | 9.196% |
| JVS70 + Common Voice70 | 140 | 30,000 | 事前固定30,000 | 9.228% |
| JVS70 + SRC4VC70 | 140 | 30,000 | 事前固定30,000 | 8.887% |

SRC4VC train 70話者・3,500発話、抽出成功3,499発話。学習可能な母音91,358区間、除外13,787区間、失敗1発話。原音声7.252時間、母音合計1.964時間。

選定したtrainの申告属性：性別{'女性': 43, '男性': 27}、年齢21–68歳。

- 同じJVS validation 15話者・各母音10区間登録で測る、母音別EERの平均。小さいほど良い。
- SRC4VC版とCommon Voice版の差は -0.340 percentage points。1 seedでの観測値で、統計的な優位性は未検証。
- 発話全体を集約したtest FAR/FRRは今回は測定していない。過去の発話単位評価との数値の直接比較はできない。
- SRC4VC予約30話者は区間抽出・特徴統計・学習・validationに使っていない。JVSのtrain/validation/test分割も保持した。
- 追加データの量、話し方、収録条件、特徴統計が同時に変わるため、年齢・性別の偏りだけを切り分ける実験ではない。
- 追加コーパスとJVSの実在人物の重複は未確認。異なる話者ラベルだけでは同一人物でないことを保証できない。

encoderは65,920 parameters・128次元、1 seed・1新規条件。同一初期化・同一scheduleで30,000 updateを完了した。

JVS各話者のbatch選択回数は2142–2143回。現行JVSモデルとの差は最大1回。保存と再読み込みで8区間のembeddingがbit単位で一致した。

[SRC4VC公式配布](https://y-saito.sakura.ne.jp/sython/Corpus/SRC4VC/index.html)。音声と重みは`artifacts/phoneme-src4vc-training/src4vc-20261009-v1`に保存し、Gitには含めない。
