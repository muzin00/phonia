# Phase 3採用encoderの学習曲線とvalidation診断

[Issue #35](https://github.com/muzin00/phonia/issues/35)。採用済み構成
`log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel`を固定して評価した。
10/25/50話者×3 seedの追加9 runはすべて完了し、失敗・資源上限による打切りは0件。
70話者の3 runは既存結果を再利用した。候補・閾値は変更せず、test splitは使用していない。

## 学習話者数とvalidation性能

主条件は未知話者のverification、登録10区間、5母音等重みのmacro EER。
全cohortでtrain特徴統計だけを変更し、最大30,000 update、早期停止規則、固定validation trial、seedを揃えた。

| 学習話者数 | 3 seed平均macro EER | seed間標準偏差 | cross-text EER | 30–49 ms EER |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 16.750% | 0.134% | 18.061% | 19.247% |
| 25 | 12.195% | 0.123% | 13.603% | 15.032% |
| 50 | 10.495% | 0.075% | 11.748% | 13.501% |
| 70 | 9.436% | 0.170% | 10.861% | 12.656% |

追加9 runのseed別EERは10話者が16.618/16.933/16.699%、25話者が12.258/12.024/12.304%、
50話者が10.558/10.391/10.537%（seed 20260926/27/28の順）。
各runの選択update、母音別EER、経過時間は`learning-curve-results.json`に保存した。
3 seedは同じ15 validation話者を使用するため、標準偏差は独立した評価話者集合に対する不確実性ではない。
この曲線は70話者で選択した構成に条件付けた診断で、別構成の学習曲線との優劣を示さない。

追加学習の記録時間は10話者で各19.1分、25話者で19.8～21.4分、50話者で19.3～26.0分。
9 runの記録時間合計は約191.1分、4並列の実経過時間は約60.2分。
観測最大RSSは1.41 GiB。最初の起動は実行監視の権限不足で中断し、孤立した学習プロセスを停止してから
9 runを最初から再実行した。成功したrunの学習中断・再開はない。

## 品質層

既存54 checkpointの保存済みvalidation指標から、無声化・品質フラグ・長母音属性を集計した。
採用構成3 seedでは無声化なしのmacro EERが8.929%。無声化ありは/i/と/u/だけに存在し、
各295/294 query、母音別EERは35.601/34.548%だった。/a/・/e/・/o/がないため、
無声化ありの5母音macro EERは定義しない。全30,533 verification queryで品質フラグは空、
長母音属性はfalseで、これらの属性の効果は比較できない。
対象3属性には欠損がなく、unknownは0件。欠損時にunknownとする既存評価規則は保持した。
全54 runの層別・母音別件数と値は`quality-strata-results.json`に保存した。

## 境界・音量感度

採用済み70話者の3 checkpointについて、validationの登録profileと母音別閾値を固定し、
照合queryだけを変更した。境界は元WAV上の区間を平行移動、音量はcrop後・DC除去前に変換した。
境界±5/10 msで有効な共通queryは30,533/30,533件、範囲外除外は0件。gainはclipしない。

| query条件 | 3 seed平均macro EER | 元条件との差 | FAR 1%閾値でのFAR | 同FRR |
| --- | ---: | ---: | ---: | ---: |
| 元条件 | 9.436% | 0.000ポイント | 0.999% | 39.395% |
| 境界 -10 ms | 8.854% | -0.582ポイント | 0.981% | 37.205% |
| 境界 -5 ms | 9.064% | -0.372ポイント | 0.986% | 38.072% |
| 境界 +5 ms | 10.015% | +0.579ポイント | 1.001% | 41.627% |
| 境界 +10 ms | 10.833% | +1.397ポイント | 0.983% | 44.758% |
| gain -6 dB | 11.665% | +2.229ポイント | 0.879% | 57.940% |
| gain +6 dB | 13.600% | +4.164ポイント | 1.284% | 52.373% |

固定閾値でのFRRは元条件でも約39%。音量変化ではEERとFRRが大きく悪化した。
この結果は同一validation話者・収録データへの人工的な変換に限られ、実環境での性能や
別収録機器への一般化を保証しない。母音別・seed別のEER、FAR、FRRは`sensitivity-results.json`、
各条件のscoreと指標は`sensitivity/<seed>/`に保存した。

## 成果物、再現手順、最終testへの引継ぎ

成果物ルート（Git管理外）:
`artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2/`。
`plan.json`に元の採用記録・予算・実装・cohort別特徴統計のchecksumを固定した。
SHA-256はplan `356cbbdbd1007c1e16494cbdf44f71c58ed4b6ab0d2d632122075cebfccf32e7`、
学習曲線集計 `0a460f9d5245c1b5958a76ad88503ca105c9782005263b884667f86dcba32ec4`、
品質層集計 `9dd01181dee5c9b0088c7b3094993a56e16046ba2d8db98f21a4670f526b0c47`、
感度集計 `f876e88f561c3d78b59423ea7209e0b0b5b73e7505b5881b577f4164801aeea5`。
run別`execution.json`、学習summary、checkpoint、選択validation指標とscoreを保持した。

```sh
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-speaker-encoder/scripts/run_learning_curve.py report
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-speaker-encoder/scripts/build_learning_curve_report.py
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-speaker-encoder/scripts/build_quality_strata_report.py
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-speaker-encoder/scripts/build_sensitivity_report.py
```

学習の計画生成・実行は`run_learning_curve.py plan`と`run_learning_curve.py run`、
感度scoreの生成は`evaluate_sensitivity.py --seed <seed>`。
既存成果物との照合後に完了runを飛ばすが、再実行には学習時間と保存容量が必要。

最終testに渡す正本は既存の
`artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2/selection-evaluation/selected-bundle/<seed>/`
にある3 checkpoint、run設定、train特徴統計、seed別の母音閾値である。
checksumは[70話者比較結果](phase3-70spk-comparison-results.md)に記録済み。
最終testではこの採用構成・3 seed・登録10区間の評価手順を固定して一度だけ評価する。
今回の診断を根拠に構成や閾値を変更しない。
