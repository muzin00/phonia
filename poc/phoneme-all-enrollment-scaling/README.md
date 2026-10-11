# 全音素モデルを固定した登録区間数10・20・30比較

[全36音素モデル](../phoneme-all-training/README.md)の学習済みencoderを固定する。
登録はユーザー音声から音素別の声紋を作る処理であり、今回モデルを再学習しない。
設定は[config/protocol.json](config/protocol.json)の1ファイル。

## 実測結果

登録上限10・20・30で、test通常文EERは1.769→1.361→1.497%、
別テキストは1.531→1.221→1.020%だった。20・30とも10より低い観測値となった。
通常文では20が最良で、20→30は単調な改善になっていない。
本人平均スコアと本人−他人の平均差は両testで登録上限に伴って増加した。
validationも通常文0.787→0.680→0.680%、別テキスト0.765→0.528→0.510%となった。

30−10の対応付き話者bootstrap95%区間は、通常文[-0.714, +0.408] pp、
別テキスト[-1.050, 0.000] pp。どちらも0を含み、改善の確証には至っていない。
FAR1%校正時のtest通常文FRRは10・30とも4.400%、別テキストは15.556→14.222%だった。
登録に使用した切片の合計時間中央値は、testで26.610→45.740→59.460秒/人。
これは選択した切片の内部利用時間で、必要な録音時間ではない。

再学習なしで登録を増やす効果が見られたが、全指標が一律・単調に改善する結果ではなかった。
全108,000 trial、12 EER、144 FAR/FRR値を独立再計算し、
共有話者bootstrapの240 EER値も照合した。固定入力・モデル・元音声など6,869ファイルをhash検証した。
HTMLの12・6・74行の3表は、JSONとMarkdownの値との一致を確認した。

## 比較条件

- 各音素の登録区間数の上限10・20・30を全て測定する。
- 元と同じ登録候補音声・選択seedで、録音元を分散した区間を選ぶ。10の区間は20・30にも含まれる。
- 全36学習音素を残す。希少音素は実在する区間だけを使い、複製・paddingで上限を満たさない。tyは未学習。
- 各splitで全話者に登録できる音素の集合、照合発話、母音不足の扱い、切片、前処理、等重み統合は前回と共通。
- 登録上限ごとにvalidation通常文で閾値を校正し、全条件の閾値を固定してからtestを採点する。testで条件を選択しない。
- 通常文750発話中735・別テキスト450発話中392を全条件共通にEER評価する。母音不足も全入力FAR/FRRに含める。
- 本人・他人スコアの平均・標準偏差・5/95パーセンタイル、平均差、EER、FAR/FRR、音素別実登録数、利用音声量を保存する。
- 既存15 test話者を共有する2,000回の対応付き話者bootstrapで、20−10・30−10・30−20のEER差を測定する。

登録上限10のprofile・全36,000 trial・embedding・validation閾値・指標は前回と完全一致を要求する。
登録・照合は原音声のpathだけでなくSHAでも分離を検査する。
新しい学習データ・音声抽出・人手レビュー回答は追加しない。
元音声、alignment、特徴cache、モデル、依存コード、全条件の登録入力を採点前にhash固定する。
保存スコアをfloat64で独立再構成し、EER・FAR/FRRとbootstrap抽出例を別計算で照合する。
元の評価は既に観測済みのJVS testであり、未観測話者での確認ではない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-all-enrollment-scaling/tests -v
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-enrollment-scaling/study.py prepare
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-enrollment-scaling/study.py evaluate
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-enrollment-scaling/study.py audit
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-enrollment-scaling/report.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-enrollment-scaling/study.py verify
```

prepareは未使用runディレクトリにのみ実行する。実測成果物はignoredな`artifacts/`配下へ保存し、
公開する数値・表は[HTML](evaluation-results.html)・[Markdown](evaluation-results.md)・
[JSON](evaluation-results.json)・[解釈](interpretation.md)に保存する。
