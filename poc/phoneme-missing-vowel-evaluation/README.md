# 照合時の母音不足への対応

[Issue #47](https://github.com/muzin00/phonia/issues/47) の2番目の検証。現行の5母音必須方式に、取得できた母音だけを等重みで統合する4母音以上・3母音以上の方式を比較する。

2026-10-08に実測を完了した。4母音以上では、主条件の全入力FRRが通常文4.533→2.533%、別テキスト14.667→3.333%へ低下した。全入力FARは通常文0.714→0.733%、別テキスト0.730→0.778%へ上昇した。本人救済と他人受入の変化、3母音の小標本の限界は[結果の解釈](interpretation.md)、全条件とCIは[測定表](evaluation-results.md)と[HTML](evaluation-results.html)を参照する。

## 比較条件

- 固定した境界内母音encoder（65,920 parameters、128次元）と各母音10区間の登録profileを使用。登録数の拡大と組み合わせない。
- 同一の発話全区間・利用可能母音区間を使用。`strict5`、`available4`、`available3` の3方式。母音内のsegment cosineを平均し、存在する母音のscoreを等重み平均する。欠けた母音を0として5で割る処理は行わない。
- 最低母音数に満たなければ `no_score` とし、本人拒否と全入力分母に残す。品質除外、重複・不正区間、登録／照合音声の重複やencoderの不一致に対する検査は引き継ぐ。
- 各方式についてvalidation通常文をまとめて目標FAR 1%、0.1%、EER動作点の閾値を校正し、別テキストとtestへ固定適用する。母音数・欠損パターン別の閾値は作らない。通常文validationの母音不足は4母音15発話だけで、3母音の校正データはない。
- 主指標は全入力FRR、条件付き／全入力FAR、coverage。EER、実件数、欠損パターン別の結果、不足発話の救済と既存発話の悪化を併記する。scoreを出せたことだけでは改善と扱わない。
- 共有10,000回の話者bootstrapで固定閾値でのCIと方式差を求める。最低母音数による方式差以外を固定する。

既にPhase 3/6/7と登録区間数検証で観測済みのJVS testを使う探索的な追加検証。欠損は元発話の自然な母音不足・既存の区間適格性によるもので、人為的な母音削除や品質重み、Transformerの学習は行わない。未観測の独立holdoutは別途必要。

## 実行

Phase 6のvenvと既存のJVS・固定encoder・Phase 6/7・登録区間数検証の成果物を使う。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-missing-vowel-evaluation/matplotlib-cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-missing-vowel-evaluation/run_study.py \
  --output-dir artifacts/phoneme-missing-vowel-evaluation/missing-vowels-20261008-v1
```

実行前にコード・入力・固定成果物をhashで凍結する。validation → 閾値とbootstrapの凍結 → test → 件数・CI・レポート検査の順に実行。各embeddingは3回反復して一致を要求する。5母音揃う発話の各母音scoreと統合scoreは全方式で元の登録10条件に一致を要求し、`strict5` のprofile・全trial・閾値・指標・CIも既存結果と一致を要求する。

元のPhase 3/6/7と登録区間数検証のコード・設定・成果物は保持し、今回の実験はこの独立ディレクトリに追加する。生score・embedding・凍結記録は `artifacts/phoneme-missing-vowel-evaluation/` に保存する。Phase 8全体の完了や本番照合APIの切り替えを意味しない。
