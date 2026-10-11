# 録音品質と照合性能の切り分け

JVSとCommon Voiceの録音品質が異なるという聴感上の観察を、既存データの記述的比較と同一音声への介入で検証する。既に観測済みのvalidation/testを開発・診断用として扱い、新しい独立性能保証や採用判定には使わない。既存モデル・元音声・登録profile・受付方針は変更しない。

## 固定した比較

- 品質比較: 学習集合はJVS/CV各70話者から固定hash順10発話/話者。評価集合はJVSの通常文照合全1,500発話、CVの照合全1,800発話。登録音声も別集計する。双方ともモデルへ渡す24kHz PCMで測定するため、CVの元MP3変換の影響も含む。
- 同一音声の介入: JVSのvalidation/test各15話者から通常文10発話/話者をhash順に選ぶ。登録はclean固定。照合全波形に、clean、−12dB、白色雑音20/10dB、300〜3400Hz帯域制限、合成残響RT60設定0.3秒を個別に適用する。noise seedは発話に固定し、20/10dBで同じ雑音を使う。
- 音素境界固定・支持音素固定（fixed）では元から採用された区間を保持し、encoderへの波形変化だけを見る。fixed_qcでは同じ境界で元のRMS基準を再適用する。realignedでは加工済み全波形をJuliusで再抽出し、同じ30〜250ms中心cropとRMS基準を適用する。原文/G2Pは固定。
- 5母音必須を主診断、4母音以上を補助として同じ全試行に適用する。採点不能・alignment失敗も本人拒否に残す。元JVS閾値での全入力FAR/FRR、採点可能入力のEER、本人・他人のscore分布を比較する。条件ごとのvalidation FAR1%校正も補助報告し、test加工前に凍結する。
- CVの品質別比較ではvalidation照合音声で三分位境界を固定してtestへ適用する。音量・energy contrast・4kHz超の相対power・有効音声長を個別に層別する。話者と文長の交絡が残るため因果効果とは扱わない。

## 音響指標の意味と限界

20ms非重複フレームのRMSを用いる。energy contrastはフレームRMSのP90−P10（dB）。quiet RMSは下位10%フレームのpower平均、active RMS/秒数はRMSがmax(P90−25dB, −50dBFS)を超えるフレームから計算する。これはVADの正解ラベルや真のSNRではなく、無音処理・話し方・語音にも依存する。値が高いだけで高品質とは断定しない。クリッピング近傍は|PCM|≥0.999、帯域はWelch powerの95%累積周波数と4kHz超の比率で記述する。残響時間を自然音声から推定することはしない。

白色雑音SNRは元音声のactive RMSに対して設定し、付加前後の実測値も保存する。帯域制限は6次Butterworthの前後方向filter、残響は固定乱数による指数減衰IR（direct/reverberant energy比0dB、0.3秒で振幅−60dB、末尾は元長に切る）を使用。帯域制限/残響は全波形RMSを元に揃え、単なる音量差を抑える。全加工後PCM16量子化・クリッピング率を記録する。これらはCVの実際の劣化過程の再現ではない。

## 実行

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-quality/study.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-quality/study.py quality
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-quality/study.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-quality/study.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-quality/report.py
```

コード・設定・入力hashをfreezeで固定。途中再開時は保存済み加工結果のhashと設計を検証して再利用する。新しい研究条件では別runを使う。
