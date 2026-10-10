# Phase 8: 30,000・45,000・60,000回の学習傾向を比較する

[Issue #47](https://github.com/muzin00/phonia/issues/47)の学習予算検証。
Common Voice版とSRC4VC版を各1 seed、初期化から60,000 updateまで学習する。
各学習内の30,000・45,000・60,000 updateを固定して比較する。
2026-10-10に両学習と、事前指定した6個のcheckpointのvalidation・発話単位test評価まで完了した。

Common Voiceのtest FARは30,000→60,000回で通常文2.305→2.038%、別テキスト1.794→1.571%となった。
SRC4VCの通常文の発話EERは1.224→1.098→0.962%と低下したが、別テキストFRRは13.778→14.222%に増えた。
主比較12指標の差の95% CIは全て0を含む。改善の兆候は一部にあるが、総合的な優位性や大きな向上は確認できていない。
単一母音区間のvalidation EERは30,000回以降ほぼ横ばいだった。

[発話単位HTML比較表](evaluation-results.html)、[結果の解釈](interpretation.md)、
[学習曲線HTML](training-results.html)、[全条件CSV](all-conditions.csv)、[対応付き差CSV](paired-differences.csv)を参照する。

## 固定条件

学習条件は[protocol.json](config/protocol.json)の1組、評価条件は[evaluation.json](config/evaluation.json)の1組。
新規の学習は2本だけ。45,000回用の別学習や、学習率・seedの探索は行わない。

- JVS70＋Common Voice70：既存の522,359母音区間とtrain由来の特徴統計を再利用する。
- JVS70＋SRC4VC70：既存の415,271母音区間とtrain由来の特徴統計を再利用する。
- 両モデルで65,920 parametersの128次元encoder、140ラベル分類head、初期encoder・head、seed 20260926、batch・損失・前処理・optimizerを揃える。
- 過去の30,000回用設定から変更する学習パラメータは最大updateの60,000だけ。warmupは1,000回で、cosine decayの終点が60,000になる。
- early stoppingは無効。1,000 updateごとのvalidationは診断用で、採用時点を変更しない。
- 30,000・45,000・60,000 updateでoptimizer・scheduler・RNGを含むcheckpointと推論bundleを保存する。
- snapshotの保存・再読み込み確認ではRNGを復元し、後続の学習への影響を防ぐ。
- JVSのtrain/validation/test 70/15/15を保持し、SRC4VC予約30話者は今回も使用しない。
- 音声・重み・全試行スコアはGit管理外の`artifacts/phoneme-training-budget/budget-20261010-v1/`へ保存する。

過去の30,000回時点は学習率の減衰期間が異なるため、新しい曲線の途中checkpointの代わりにはしない。
各コーパス内の30,000→60,000を主比較、30,000→45,000を補助の傾向確認とする。
両コーパスのデータ量・属性・収録条件・特徴統計は異なり、コーパス間の差からデータ量の効果だけを特定しない。

## 発話単位評価

学習を完了し、6個のcheckpointを全て固定してから評価する。
同じJVS validation/test各15話者、通常文750・別テキスト450発話、各発話15申告話者との18,000試行を使う。
登録は各母音10区間、照合は発話中の利用可能な母音区間を全て使用し、5母音を必須にする。
母音内cosine平均を5母音で等重み平均し、score不足は拒否として全入力FAR/FRRの分母に残す。
発話EERはscoreが得られた発話だけを対象とし、score coverageと併記する。
同じvalidation目標FARで校正する比較であり、testの実測FARを一致させる比較ではない。

全6モデルのvalidation推論・閾値決定を終えてから、一度に固定した閾値をtestへ適用する。
目標FAR 1%を主動作点、0.1%とEER動作点を補助とする。testによる閾値の再調整・checkpointの選別は行わない。
全6時点のtest評価は学習前に指定しており、validationの結果によって評価対象を変更しない。
各embeddingを3回計算してbit単位の一致を確認し、公開の登録・照合APIとのcacheなし照合も行う。
現行JVS70・15,000 updateをchecksum検証付きの参考値として再利用する。
全条件で同じ10,000回・seed 20260929の話者bootstrapを使い、対応付き差の95% CIを計算する。

これは1 seed・既に観測したJVS testの探索的な検証で、新しい独立holdoutではない。
CIは固定モデル・固定閾値の話者変動を表し、学習seedの変動や閾値校正の不確かさは含まない。
同じschedule内で学習回数と学習率が一緒に推移する曲線であり、予算ごとに最適化した学習設定の比較ではない。
今回増やすのは学習回数で、データ量は増やさない。データ量のスケーリング則を直接測る実験ではない。
コーパス間の実在人物重複は未確認である。

## 再現手順

既存のデータ準備・特徴統計を保持した環境で実行する。凍結入力のchecksumが異なる場合は停止する。
CPU 1 thread・workers 0、既存のverification環境を使用する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-budget/budget_training.py prepare
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-budget/budget_training.py train --corpus cv
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-budget/budget_training.py train --corpus src
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-budget/run_evaluation.py training-report
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-budget/run_evaluation.py evaluate --output-dir artifacts/phoneme-training-budget/budget-20261010-v1/evaluation
```

学習は1,000 updateごとの`last.pt`からoptimizer・scheduler・sampler・RNGを復元できる。
完了後の学習コマンドは入力・成果物のchecksumを再検証して終了する。
評価には未使用の出力directoryが必要。既存の評価成果物は上書きしない。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-training-budget/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check --ignore E402 poc/phoneme-training-budget
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-training-budget
```
