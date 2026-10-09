# Phase 8: SRC4VC版の発話単位評価

[Issue #47](https://github.com/muzin00/phonia/issues/47)の追加検証。
[SRC4VCでの学習](../phoneme-src4vc-training/README.md)の30,000更新checkpointを固定して評価する。
validation・閾値固定・test・bootstrap・独立監査・比較表生成まで完了した。
主条件の全入力FARはCV→SRC4VCで通常文2.657→1.714%、別テキスト2.000→1.413%。
全入力FRRは通常文2.933→2.400%、別テキスト13.556→13.778%。差の95% CIはいずれも0を含む。
[比較表HTML](evaluation-results.html)と[結果の解釈](interpretation.md)を参照する。
条件は[protocol.json](config/protocol.json)の1つ。新規学習や登録数・母音不足対応の探索は行わない。

## 比較条件

- 主比較：JVS70＋SRC4VC70・30,000更新 − JVS70＋Common Voice70・30,000更新。
- 参考比較：SRC4VC版 − 現行JVS70・validation best 15,000更新。
- 登録は各母音10区間、照合は発話中の利用可能な母音区間全てを使用し、5母音を必須にする。
- 同じJVS validation/test各15話者、通常文750・別テキスト450発話、各発話15申告話者との18,000試行を使用する。
- 母音内cosineを平均してから、5母音を等重み平均する。score不足は拒否し、全入力FAR/FRRの分母に残す。
- 各モデルのvalidation通常文で目標FAR 1%・0.1%・EER動作点の閾値を決め、test推論前に固定する。testによる再校正はしない。
- 既存JVS/CV版の推論と閾値をchecksum検証して再利用する。既存指標・95% CIが一致することも完了条件にする。
- SRC4VC版の各区間embeddingは3回計算し、bit単位の一致を確認する。公開の登録・照合APIとの照合も行う。
- 10,000回・seed 20260929の同じ話者bootstrapを全モデルで共有し、対応付き差の95% CIを計算する。

事前に観測したJVS testを使う1 seedの探索的な評価。CIは固定モデル・固定閾値での話者変動を表し、学習seedの変動を表さない。
SRC4VCとCommon Voiceの追加区間数は91,358と198,446で異なり、話し方・収録条件・特徴統計も同時に変わる。
属性の偏りだけを切り分ける実験ではない。JVSモデルだけはcheckpoint採用規則が異なる。
SRC4VC予約30話者での評価・新しい独立holdoutはこの比較の対象外。コーパス間の実在人物の重複は未確認。

## 実行

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-src4vc-evaluation/run_evaluation.py --output-dir artifacts/phoneme-src4vc-evaluation/evaluation-20261009-v1
```

入力・コード・重みを推論前に固定し、validation推論 → 閾値固定 → test推論 → 指標・bootstrap → 独立監査 → 比較表生成の順に実行する。
音声・重み・全試行スコアはGit管理外の`artifacts/`へ保存する。
表形式の[HTML](evaluation-results.html)、[Markdown](evaluation-results.md)、[CSV](all-conditions.csv)、[対応付き差](paired-differences.csv)を生成する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-src4vc-evaluation/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check --ignore E402 poc/phoneme-src4vc-evaluation
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-src4vc-evaluation
```
