# 録音品質と照合性能の診断結果

既に観測済みの話者を用いた探索的診断。FAR/FRRは採点不能を拒否に含む全入力割合、EERは採点可能入力のみ。単位は%。

## コーパスの品質指標

中央値。contrastは真のSNRではなく、20msフレームのRMS P90−P10。

| 集合 | コーパス | 音声数 | contrast dB | active RMS dBFS | quiet RMS dBFS | active秒 | 95%帯域Hz | clip近傍あり件数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test/verification | CV | 900 | 50.22 | -22.91 | -73.84 | 1.96 | 1594 | 7 |
| train/training | CV | 700 | 52.53 | -21.61 | -74.77 | 2.42 | 1406 | 11 |
| validation/verification | CV | 900 | 50.24 | -22.34 | -72.45 | 2.38 | 1195 | 10 |
| test/verification | JVS | 750 | 51.63 | -26.60 | -75.78 | 4.77 | 1430 | 0 |
| train/training | JVS | 700 | 51.49 | -25.59 | -75.53 | 4.36 | 1336 | 0 |
| validation/verification | JVS | 750 | 52.49 | -25.31 | -77.23 | 4.72 | 1383 | 0 |

## 同一JVS音声への加工

各条件test150発話・本人150試行/他人2,100試行。登録clean固定。元JVS閾値と加工条件別validation校正を比較。

| 加工 | 境界/QC | 受付 | coverage | EER | 元閾値FAR | 元閾値FRR | 条件校正FAR | 条件校正FRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | fixed | vowels5 | 97.333 | 1.370 | 0.286 | 4.667 | 0.333 | 4.667 |
| clean | fixed | vowels4 | 100.000 | 1.333 | 0.286 | 2.000 | 0.333 | 2.000 |
| clean | fixed_qc | vowels5 | 97.333 | 1.370 | 0.286 | 4.667 | 0.333 | 4.667 |
| clean | fixed_qc | vowels4 | 100.000 | 1.333 | 0.286 | 2.000 | 0.333 | 2.000 |
| clean | realigned | vowels5 | 97.333 | 1.370 | 0.286 | 4.667 | 0.333 | 4.667 |
| clean | realigned | vowels4 | 100.000 | 1.333 | 0.286 | 2.000 | 0.333 | 2.000 |
| gain-minus12 | fixed | vowels5 | 97.333 | 4.795 | 0.000 | 29.333 | 0.048 | 22.667 |
| gain-minus12 | fixed | vowels4 | 100.000 | 4.667 | 0.000 | 28.000 | 0.048 | 21.333 |
| gain-minus12 | fixed_qc | vowels5 | 96.000 | 5.159 | 0.000 | 33.333 | 0.048 | 22.000 |
| gain-minus12 | fixed_qc | vowels4 | 100.000 | 5.143 | 0.000 | 31.333 | 0.048 | 19.333 |
| gain-minus12 | realigned | vowels5 | 95.333 | 9.091 | 0.000 | 49.333 | 0.333 | 40.000 |
| gain-minus12 | realigned | vowels4 | 100.000 | 8.857 | 0.000 | 47.333 | 0.333 | 38.000 |
| white-snr20 | fixed | vowels5 | 97.333 | 13.699 | 11.143 | 18.667 | 0.810 | 54.667 |
| white-snr20 | fixed | vowels4 | 100.000 | 13.333 | 11.571 | 16.000 | 0.857 | 53.333 |
| white-snr20 | fixed_qc | vowels5 | 97.333 | 13.699 | 11.143 | 18.667 | 0.810 | 54.667 |
| white-snr20 | fixed_qc | vowels4 | 100.000 | 13.333 | 11.571 | 16.000 | 0.857 | 53.333 |
| white-snr20 | realigned | vowels5 | 97.333 | 14.335 | 11.333 | 19.333 | 0.857 | 56.000 |
| white-snr20 | realigned | vowels4 | 100.000 | 14.381 | 11.762 | 16.667 | 1.048 | 52.000 |
| white-snr10 | fixed | vowels5 | 97.333 | 30.137 | 3.095 | 82.667 | 0.952 | 99.333 |
| white-snr10 | fixed | vowels4 | 100.000 | 30.095 | 3.333 | 82.667 | 1.048 | 98.667 |
| white-snr10 | fixed_qc | vowels5 | 97.333 | 30.137 | 3.095 | 82.667 | 0.952 | 99.333 |
| white-snr10 | fixed_qc | vowels4 | 100.000 | 30.095 | 3.333 | 82.667 | 1.048 | 98.667 |
| white-snr10 | realigned | vowels5 | 86.667 | 31.538 | 2.048 | 90.000 | 0.143 | 100.000 |
| white-snr10 | realigned | vowels4 | 89.333 | 30.597 | 2.238 | 89.333 | 0.143 | 100.000 |
| band-300-3400 | fixed | vowels5 | 97.333 | 36.301 | 0.000 | 100.000 | 1.476 | 96.667 |
| band-300-3400 | fixed | vowels4 | 100.000 | 36.667 | 0.000 | 100.000 | 1.333 | 96.667 |
| band-300-3400 | fixed_qc | vowels5 | 97.333 | 36.350 | 0.000 | 100.000 | 1.333 | 96.667 |
| band-300-3400 | fixed_qc | vowels4 | 100.000 | 36.667 | 0.000 | 100.000 | 1.286 | 96.667 |
| band-300-3400 | realigned | vowels5 | 97.333 | 36.301 | 0.000 | 100.000 | 1.476 | 96.667 |
| band-300-3400 | realigned | vowels4 | 100.000 | 36.667 | 0.000 | 100.000 | 1.476 | 96.667 |
| reverb-rt60-0.3 | fixed | vowels5 | 97.333 | 2.642 | 0.429 | 8.000 | 0.476 | 7.333 |
| reverb-rt60-0.3 | fixed | vowels4 | 100.000 | 2.571 | 0.429 | 6.000 | 0.476 | 5.333 |
| reverb-rt60-0.3 | fixed_qc | vowels5 | 97.333 | 2.495 | 0.429 | 8.000 | 0.476 | 7.333 |
| reverb-rt60-0.3 | fixed_qc | vowels4 | 100.000 | 2.429 | 0.429 | 6.000 | 0.476 | 5.333 |
| reverb-rt60-0.3 | realigned | vowels5 | 97.333 | 1.370 | 0.524 | 7.333 | 0.571 | 7.333 |
| reverb-rt60-0.3 | realigned | vowels4 | 100.000 | 1.333 | 0.571 | 4.667 | 0.619 | 4.667 |

## CV testの品質別比較

境界はCV validationの三分位。閾値は前回CV校正の固定値。品質以外に話者・文長・音素構成も異なる。

| 指標 | 層 | 受付 | 発話/話者 | active秒中央値 | coverage | EER | FAR | FRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active_rms_dbfs | low | vowels5 | 340/28 | 1.94 | 65.294 | 3.573 | 0.568 | 41.176 |
| active_rms_dbfs | low | vowels4 | 340/28 | 1.94 | 90.588 | 3.571 | 0.771 | 20.000 |
| active_rms_dbfs | middle | vowels5 | 359/29 | 2.02 | 67.409 | 3.306 | 0.413 | 40.669 |
| active_rms_dbfs | middle | vowels4 | 359/29 | 2.02 | 92.201 | 3.927 | 0.644 | 20.613 |
| active_rms_dbfs | high | vowels5 | 201/22 | 1.96 | 58.209 | 5.128 | 0.274 | 55.721 |
| active_rms_dbfs | high | vowels4 | 201/22 | 1.96 | 89.552 | 5.402 | 0.412 | 34.328 |
| active_seconds | low | vowels5 | 429/30 | 1.36 | 37.296 | 3.750 | 0.305 | 69.231 |
| active_seconds | low | vowels4 | 429/30 | 1.36 | 81.119 | 4.598 | 0.627 | 35.431 |
| active_seconds | middle | vowels5 | 238/30 | 2.31 | 84.454 | 3.980 | 0.536 | 25.210 |
| active_seconds | middle | vowels4 | 238/30 | 2.31 | 100.000 | 4.202 | 0.652 | 12.605 |
| active_seconds | high | vowels5 | 233/30 | 4.04 | 94.421 | 4.091 | 0.592 | 17.597 |
| active_seconds | high | vowels4 | 233/30 | 4.04 | 100.000 | 4.292 | 0.651 | 12.446 |
| energy_contrast_db | low | vowels5 | 229/20 | 2.18 | 69.869 | 3.125 | 0.572 | 37.555 |
| energy_contrast_db | low | vowels4 | 229/20 | 2.18 | 91.266 | 3.828 | 0.858 | 20.524 |
| energy_contrast_db | middle | vowels5 | 374/26 | 1.84 | 64.171 | 3.333 | 0.526 | 43.048 |
| energy_contrast_db | middle | vowels4 | 374/26 | 1.84 | 92.246 | 3.768 | 0.701 | 20.588 |
| energy_contrast_db | high | vowels5 | 297/22 | 1.94 | 60.943 | 6.211 | 0.232 | 50.842 |
| energy_contrast_db | high | vowels4 | 297/22 | 1.94 | 89.226 | 6.103 | 0.395 | 29.293 |
| high_frequency_fraction | low | vowels5 | 174/25 | 2.03 | 71.839 | 3.200 | 0.396 | 35.057 |
| high_frequency_fraction | low | vowels4 | 174/25 | 2.03 | 93.103 | 2.469 | 0.555 | 17.816 |
| high_frequency_fraction | middle | vowels5 | 282/30 | 2.10 | 65.603 | 4.026 | 0.379 | 46.454 |
| high_frequency_fraction | middle | vowels4 | 282/30 | 2.10 | 92.553 | 4.598 | 0.489 | 23.759 |
| high_frequency_fraction | high | vowels5 | 444/27 | 1.86 | 61.036 | 4.059 | 0.497 | 46.396 |
| high_frequency_fraction | high | vowels4 | 444/27 | 1.86 | 89.189 | 4.981 | 0.769 | 25.450 |

## energy contrastと有効音声長の交差比較

| contrast | 長さ | 受付 | 発話数 | FAR | FRR |
| --- | --- | --- | ---: | ---: | ---: |
| low | short | vowels5 | 269 | 0.423 | 56.877 |
| low | short | vowels4 | 269 | 0.795 | 26.394 |
| low | long | vowels5 | 182 | 0.739 | 16.484 |
| low | long | vowels4 | 182 | 0.815 | 10.989 |
| high | short | vowels5 | 294 | 0.246 | 60.884 |
| high | short | vowels4 | 294 | 0.457 | 31.973 |
| high | long | vowels5 | 155 | 0.489 | 23.226 |
| high | long | vowels4 | 155 | 0.512 | 16.774 |

## 検証と限界

独立float64再計算 81,000行、最大誤差 3.99e-08。12条件の加工再現・モデル再推論・元PCM特徴再抽出を確認。

話者bootstrap 2,000回の区間・スコア分布・登録音声の品質・alignment変化は[JSON](evaluation-results.json)を参照。区間は固定モデル/閾値に条件付く探索的なもの。多数比較の補正はせず、自然音声の層別差を因果効果と解釈しない。合成劣化はCVの実録音を再現したものではない。新しい独立testや別日・別端末での性能保証は含まない。
