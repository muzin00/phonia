# Phase 8: 学習回数を揃える切り分け

[Issue #47](https://github.com/muzin00/phonia/issues/47)の学習データ増加に続く検証。
前回は70→140話者ラベルに増やし、同じupdate数で学習したため、採用checkpointにおけるJVS各話者の
batch選択回数が約2,143→1,071回となった。追加データの効果と、JVSの学習機会の減少が混ざっている。

2026-10-08に学習・発話単位評価まで完了した。JVS各話者の選択回数は2,142～2,143回、現行モデルとの最大差は1回だった。
追加データモデルの15,000→30,000 updateで、validation目標FAR 1%のtest全入力FARは通常文2.629→2.657%、
別テキスト1.905→2.000%となった。全入力FRRは通常文3.067→2.933%、別テキスト13.556→13.556%。
本人発話の救済は通常文1件のみで、FAR改善は確認できなかった。
[評価表](evaluation-results.md)、[HTML比較表](evaluation-results.html)、[切り分けの解釈](interpretation.md)を参照する。

新規条件は1つ、seedは20260926の1つ。前回のJVS＋Common Voiceモデルの採用15,000 updateから、
同じデータ・特徴統計・encoder・損失・batch・optimizer・scheduler・RNGを復元して30,000 updateまで続ける。
30,000 updateでJVS各話者のbatch選択回数が、現行JVSモデルの採用15,000 updateと同じ2,142～2,143回となる。
各話者の実測回数が現行と最大1回差以内であることを完了条件にする。

## 固定条件

設定は[protocol.json](config/protocol.json)の1つに固定する。
元のJVSモデル15,000 update、追加データモデル15,000 updateの2つは保存済み結果を再利用する。
新規に学習するのは追加データモデル30,000 updateだけ。

今回は学習機会を揃えた時点の比較なので、early stoppingを無効にし、評価対象を事前に30,000 updateへ固定する。
validationは途中経過の記録と発話単位評価の閾値校正に使い、途中のcheckpointへ採用対象を変更しない。
既存の最大30,000 updateのcosine scheduleとwarmup 1,000 updateは維持する。
追加学習の後半では学習率が下がるため、最適な学習率・総学習予算を検証したことにはならない。
元の2モデルはvalidationで採用したcheckpoint、今回のモデルは固定updateのcheckpointという選択規則の違いを明記する。

元の15,001～18,000 updateを再現する際は、全updateのloss・accuracy・gradient norm・学習率が
元の履歴とbit一致することを要求する。18,001 update以降も元と同じ学習計算を続ける。
元のコード・設定・データ・結果ファイルは保持し、今回の学習結果を別ディレクトリへ保存する。

発話単位評価は、同じ登録各母音10区間、5母音必須、同じ全適格母音区間、同じvalidation/test発話で行う。
各モデルのvalidation通常文で目標FAR 1%・0.1%・EER動作点の閾値を決め、test通常文・別テキストへ固定適用する。
本人のscore不足を全入力FRRの拒否に数える扱い、等重み統合、10,000回の共有話者bootstrapも引き継ぐ。
主比較は追加データモデル30,000−15,000 update。現行JVSモデルとの比較も併記する。

観測済みtestの結果から着想した探索的な切り分けであり、独立holdoutではない。
追加Common Voiceの匿名client_idとJVS評価話者の実在人物の重複は不明。
今回の結果だけで、追加データの収録条件・モデル容量・最適な学習量に関する仮説をすべて確定・否定しない。

## 実行

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-training-exposure-ablation/train_extension.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-training-exposure-ablation/train_extension.py train
```

`prepare`は未使用のrunディレクトリに入力・コード・条件を凍結する。
`train`は元の採用checkpointか、今回の`last.pt`から再開する。完了後の再実行では凍結入力と出力を検証する。
途中経過、最終checkpoint、推論用bundle、学習回数の検査結果は
`artifacts/phoneme-training-exposure-ablation/exposure-20261008-v1/`へ保存する。

学習完了後に、未使用の評価runディレクトリを指定して発話単位評価を実行する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-verification-evaluation/.venv/bin/python \
poc/phoneme-training-exposure-ablation/run_evaluation.py \
--output-dir artifacts/phoneme-training-exposure-ablation/evaluation-20261008-v1
```

既存2モデルの入力・score・閾値・評価結果をhashで検証して再利用し、30,000 updateのモデルだけを新たに推論する。
validationの閾値を凍結してからtestを推論する。共有bootstrapの95%区間を独立に再計算し、
既存2モデルの結果が前回と完全一致すること、3モデルの評価対象・score取得率が同一であることを確認する。
完了時には評価表をMarkdown・HTML・CSV・JSONで出力する。
学習延長によって解消した／新たに増えた他人誤受入と、救済した／新たに拒否した本人発話を、
モデルごとの凍結validation閾値で対応付けて集計する。母音不足の拒否も集計に含める。
全話者ペアの他人誤受入件数を`diagnostics.json`へ保存する。
これは観測した判定の変化であり、残る誤判定の原因を特定する証拠とは区別する。

完了runでは、元の3,000 updateの履歴が完全一致し、15,000件の追加履歴が連続・有限であることを確認した。
保存・再読込後の8区間のembeddingはbit一致。既存2モデルの4評価セルは、指標とCIを含めて前回と完全一致した。
36件の生scoreからの率の検査、140件のCI再計算、HTMLの6表・全条件CSV36行・対応付き差56行の整合確認を通過した。

## 検証

```sh
poc/phoneme-speaker-encoder/.venv/bin/python -m unittest discover -s poc/phoneme-training-exposure-ablation/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check --ignore E402 poc/phoneme-training-exposure-ablation
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-training-exposure-ablation
```
