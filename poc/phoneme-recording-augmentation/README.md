# 録音条件の拡張によるencoder追加学習の対照比較

音量前処理だけの比較とは分け、元の全36音素encoderと話者分類headを同一の最終checkpointから追加学習する。**cleanだけの追加学習**と**同じ原区間・更新回数で録音条件を拡張する追加学習**を3 seedで比較する。元encoderも対照に残す。原音声・登録方式・統合方式は固定。

学習候補は前回品質診断で固定した既存学習140話者×10発話から、元の採用済み音素区間のみを抽出。話者×音素ごと最大12区間を発話分散・固定hash順で選ぶ。2実区間以上ある組だけpositive pairに用い、実在しない区間で補充しない。実際にpositiveを組める音素をtrain側だけから固定する。既存の学習話者・コーパス・元境界を使用し、正式な新規コーパス採用レビューは行わない。

元の特徴統計、AAM-Softmax＋0.5×音素内SupCon、5音素group×最大10話者×2区間を維持。encoderとhead両方を更新。各seedで原区間のpair列・初期重み・更新数6,000・optimizerを共通にする。AdamW lr 1e−5、weight decay 1e−4、100更新warmup後cosine減衰、勾配norm上限5。最終更新を固定し、test/validationでcheckpointを選ばない。

拡張側は各positive pairの片方をclean、もう片方を−12dB・白色雑音20dB・白色雑音10dB・300〜3400Hz帯域制限・合成残響のどれかへ一様に変更する。全波形へ加工後、元区間を切り出して元の固定特徴統計で再計算。各原録音につき各条件1変種を固定し、学習の雑音seedは評価と分ける。加工後の追加QCや境界変更はせず、支持区間を同一に保つ。拡張条件の種類は既存testで見つけた弱点を参考にしており、この研究全体を未観測test評価とは扱わない。

JVSでは前回と同じvalidation/test各150発話・6加工条件、Common Voiceでは各900照合発話を使用。全モデルで登録profileを再構築し、元音素境界と元支持集合を固定。5母音必須が主診断、4母音以上が補助。モデル別のclean JVS validation閾値を各加工に固定適用する。CVはCV validationで別校正し、JVS閾値移植も報告。全モデル・全閾値を凍結してからtest推論する。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/train.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/train.py prepare
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/train.py train
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/evaluate.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/evaluate.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-augmentation/summarize.py
```

既存quality検証環境を使用。元データ・旧checkpoint・旧結果は上書きしない。新規の学習data追加や実運用へのモデル差し替えは行わない。
