# 全音素を1モデルで学習してEERを測る

JVS＋Common Voiceの学習可能な全36音素を、同じ共通encoderで初期値から学習する。
音素をEERで選ばず、全セットを学習前に固定する。新規学習は1モデル・1 seed。
音素集合は5母音＋追加31音素。tyは学習区間が0件のため、未学習として明記する。
SRC4VCを使わず、既に検査した約96万区間の入力をbyte単位で再利用する。
新しいコーパス・音素ホワイトリスト・人手回答ファイルは追加しない。

## 学習

既存statistics pooling＋MLP、65,920 parameters、128次元を使用する。
seed・初期重み・正規化・AdamW・AAM＋SupCon・各音素group最大10話者×2区間は前回と共通。
学習されない特徴抽出とpoolingのcacheを使い、元encoderのprojectionを実際に学習する。
原境界内30〜250ms、RMS −50dBFS以上、文脈・増幅なし。
稀な音素も残し、実際の区間ペアがある話者だけを使用する。空slotは損失から除く。

216,000更新＝30,000×36÷5。5母音モデルと各母音への学習投入を揃える。
各音素30,000 group、通常600,000区間。1話者だけが適合するdyは実区間60,000件。
各母音のサンプル選択順も元の5母音モデルと共通になる。
総学習更新とcosine scheduleの期間は増えるため、純粋な音素数だけの因果効果ではなく、
各母音への投入量を維持した全音素構成の性能を測る。
warmupは1,000更新。216,000更新の最終重みを使い、validationによる早期終了・checkpoint選択は行わない。

## 評価

前回の5母音モデル・登録元音声・validation/test入力・正規化を再利用する。
通常文750発話中735、別テキスト450発話中392を両モデルで共通にスコア評価する。
5母音必須、追加音素は発話にあり、全15登録話者にprofileがある場合に使用する。
元の母音切片を短くせず、追加音素の音声も使用する。音素間は等重みで統合する。
学習可能な全音素をencoderに使っても、各発話の照合に全音素が揃うことは要求しない。
5母音不足は全入力FAR/FRRへ含める。EERはスコアのある共通発話で算出する。
validation通常文で閾値を校正し、最終重み・閾値を固定してから新モデルのtestを測定する。
基準モデルの既存test結果はそのまま再利用する。既存JVS testであり、新しいholdoutではない。

## 再現

設定は[config/protocol.json](config/protocol.json)の1ファイル。
前回入力と基準モデルを保持した環境で実行する。prepareは未使用ディレクトリにのみ実行する。
入力・コード・基準モデルを学習開始前にhash固定し、optimizer/RNG checkpointから再開できる。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-all-training/tests -v
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/study.py prepare
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/study.py train
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/study.py evaluate
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/study.py audit
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/report.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-training/study.py verify
```

結果は[evaluation-results.html](evaluation-results.html)・[evaluation-results.md](evaluation-results.md)・
[interpretation.md](interpretation.md)へ保存する。
