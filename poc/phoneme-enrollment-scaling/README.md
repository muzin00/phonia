# 母音の登録区間数を増やす実測

[Issue #47](https://github.com/muzin00/phonia/issues/47) の最初の検証。各母音の登録区間数を10・20・30に増やし、固定encoderで本人照合が改善するかを測る。Phase 8全体の完了を意味しない。

2026-10-08に実測を完了した。目標FAR 1%の主条件では10→30で全入力FRRは通常文4.533%、別テキスト14.667%のままだった。FAR/EERの観測値は低下した。目標FAR 0.1%では全入力FRRが通常文14.533→12.400%、別テキスト28.444→24.222%へ低下したが、実測FARは増えた。[結果の解釈](interpretation.md)と[測定表](evaluation-results.md)を参照する。

## 固定した比較条件

- Phase 3の選定済み `statistics_mlp`（65,920 parameters、128次元、seed 20260926）。再学習と特徴統計の更新は行わない。
- 境界内母音 `vowel_exact`、照合は同一の発話全区間内にある利用可能母音。登録は母音ごとのembedding平均を正規化し、照合は母音内cosine平均の5母音等重み平均。
- 登録10 ⊂ 20 ⊂ 30。既存seed 20260930と元WAVのround robin順を引き継ぎ、登録数以外を変えない。
- 学習70・validation 15・test 15話者は互いに別。登録用・照合用の元WAVとその内容hashの重複を禁止。
- 各splitの照合は通常文750、別テキスト450発話。5母音不足は `no_score` とし、全入力の本人拒否に含める。
- 登録数ごとにvalidation通常文だけで目標FAR 1%、0.1%、EER動作点を校正し、別テキストとtestには固定適用。testで閾値を調整しない。
- 条件付きFAR/FRR、全入力FAR/FRR、pooled EER、coverage、実件数を併記。FARの目標値とtestで観測したFARは異なる。
- 10,000回の共有話者bootstrap（seed 20260929）でCIと登録数間のpaired差を算出。本人trialは話者出現数、他人trialは照合話者×登録話者の出現数で重み付け。CIはモデルと閾値を固定した条件での値。

既にPhase 3/6/7で観測したJVS testを再利用するため、これは探索的な追加検証。未観測の独立holdoutや別日・別端末での評価ではない。学習データ増加、子音追加、欠損母音の統合改善、Transformerは今回の範囲外。

## 実行

Phase 6の既存venvとデータ・固定モデル・Phase 6/7成果物を使用する。外部モデルの取得は不要。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-enrollment-scaling/matplotlib-cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-enrollment-scaling/run_study.py \
  --output-dir artifacts/phoneme-enrollment-scaling/enrollment-20261008-v1
```

出力先は未使用のディレクトリを指定する。失敗した実行も保持し、再実行は別の出力先を使う。

実行前にコード・モデル・統計・音声・manifestと条件をhashで凍結する。validation推論、閾値・bootstrapの凍結、test推論、全件検査、レポート生成の順に進む。各embeddingは3回反復しbit単位で一致を要求する。登録10の全profile・score・閾値・test指標・CIはPhase 6/7と完全一致を要求する。

全108,000 trial、生embedding、登録音声量、score、閾値、bootstrapと凍結記録は `artifacts/phoneme-enrollment-scaling/` に保存する。36動作点の実件数・率、156個のCIとpaired差を独立に検算する。公開APIとの照合を抽出検査する。

実行成功時に、このディレクトリへ[Markdown結果](evaluation-results.md)、[HTML比較表](evaluation-results.html)、全条件CSV、paired差CSV、PNG/PDFグラフとhash manifestをコピーする。
