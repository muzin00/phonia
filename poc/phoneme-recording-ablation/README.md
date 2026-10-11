# 録音条件拡張の単独効果と除去効果

前回の混合拡張は白色雑音の識別を改善したが、clean・残響・Common Voiceの改善にはつながらなかった。今回は原因を切り分ける開発用の比較である。既に観測したvalidation/testを使うため、testという保存名でも未使用評価とは扱わない。

学習区間31,671件、元encoderと分類head、原区間pair列、3 seed、各6,000更新、optimizer、6種類の波形特徴、登録方式、元境界・支持集合を前回と同一にする。検証済み特徴を再使用し、clean/mixedの対照モデルはcheckpointを再使用する。評価は全37モデルで再計算する。/dy/ はこの学習subsetにpositive pairがなく追加学習対象外。

12 armを結果を見る前に固定する。

- `clean`: 加工なし。`augmented`: 従来の5条件混合（各pairの片方をclean、もう片方を5条件から一様選択）。
- `single1`〜`single5`: pairの片方をclean、もう片方をそれぞれ−12dB、白色雑音20dB、白色雑音10dB、300〜3400Hz帯域制限、合成残響RT60=0.3sに固定。加工率50%。
- `drop1`〜`drop5`: 混合の元スケジュールで対象条件だけcleanへ戻す。他条件の区間・回数・順序を完全に維持し、再分配しない。加工率は約40%。

単独条件は同じ総加工率で比べる。除去実験は混合内のその条件の寄与を調べる。単独条件では対象条件の露出が混合の約5倍になるため、単独結果から混合内の寄与を直接断定しない。除去による変化も他条件との相互作用を含む。この実験で加工率の最適値までは決めない。

JVS validation/test各150発話×6条件、CV各900発話。各モデルのclean JVS validationでFAR1%閾値を校正し全JVS加工に固定、CVは別校正。全モデル・閾値を凍結してから観測済みtestを再計算する。全seed・全条件を報告し、5母音必須を主評価、4母音以上を補助評価とする。FAR/FRRは未採点を含む全入力、EERは採点可能な入力。話者を対応付きで2,000回再抽出する区間は固定3 seed平均の不確実性であり、学習seed全般への保証ではない。多重比較の未補正区間は探索的な記述として扱う。

未使用データへ進める候補のスクリーニングも先に固定する。clean追加学習比で、両受付方針とも3 seed平均のclean JVS FAR増分≦0.1pp・FRR増分≦1pp、残響FRR増分≦1pp、CV FAR増分≦0.1pp・FRR増分≦1pp、かつ20dB/10dB雑音EERの両方が改善すること。これは候補を絞る点推定の基準であり、安全性や実運用採用の基準ではない。通過候補がなければ未使用データを消費しない。通過候補があれば候補・校正方法を固定して別評価する。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-recording-ablation/tests
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/train.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/train.py train
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/evaluate.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/evaluate.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/summarize.py
```

元音声・既存checkpoint・既存結果は上書きせず、成果物を専用artifactへ保存する。音声/checkpointは公開しない。音素境界を固定したencoder診断であり、再alignment失敗を含む実環境のend-to-end性能ではない。
