# 録音条件の単独・除去実験

観測済みデータの開発用比較。3 seed平均（ensembleではない）。各モデルのvalidation閾値を固定。EER/FAR/FRRの単位%。

| 受付 | 評価条件 | 学習arm | EER | FAR | FRR |
| --- | --- | --- | ---: | ---: | ---: |
| vowels5 | clean | original | 1.370 | 0.333 | 4.667 |
| vowels5 | clean | clean | 0.962 | 0.190 | 4.667 |
| vowels5 | clean | augmented | 1.631 | 0.714 | 6.667 |
| vowels5 | clean | single1 | 0.685 | 0.286 | 4.000 |
| vowels5 | clean | single2 | 1.875 | 0.778 | 6.667 |
| vowels5 | clean | single3 | 1.272 | 0.683 | 5.333 |
| vowels5 | clean | single4 | 0.685 | 0.730 | 3.333 |
| vowels5 | clean | single5 | 1.158 | 0.238 | 4.667 |
| vowels5 | clean | drop1 | 1.778 | 0.968 | 6.667 |
| vowels5 | clean | drop2 | 1.370 | 0.667 | 5.778 |
| vowels5 | clean | drop3 | 1.468 | 0.667 | 6.222 |
| vowels5 | clean | drop4 | 0.766 | 0.159 | 6.667 |
| vowels5 | clean | drop5 | 1.533 | 0.730 | 6.667 |
| vowels4 | clean | original | 1.333 | 0.333 | 2.000 |
| vowels4 | clean | clean | 0.937 | 0.190 | 2.000 |
| vowels4 | clean | augmented | 1.587 | 0.714 | 4.000 |
| vowels4 | clean | single1 | 0.667 | 0.286 | 1.333 |
| vowels4 | clean | single2 | 1.825 | 0.778 | 4.000 |
| vowels4 | clean | single3 | 1.238 | 0.683 | 2.667 |
| vowels4 | clean | single4 | 0.667 | 0.730 | 0.667 |
| vowels4 | clean | single5 | 1.127 | 0.238 | 2.000 |
| vowels4 | clean | drop1 | 1.730 | 0.968 | 4.000 |
| vowels4 | clean | drop2 | 1.333 | 0.667 | 3.111 |
| vowels4 | clean | drop3 | 1.429 | 0.667 | 3.556 |
| vowels4 | clean | drop4 | 0.746 | 0.159 | 4.000 |
| vowels4 | clean | drop5 | 1.492 | 0.730 | 4.000 |
| vowels5 | gain-minus12 | original | 4.795 | 0.000 | 28.667 |
| vowels5 | gain-minus12 | clean | 4.387 | 0.000 | 36.000 |
| vowels5 | gain-minus12 | augmented | 4.093 | 1.698 | 8.667 |
| vowels5 | gain-minus12 | single1 | 2.055 | 0.048 | 9.778 |
| vowels5 | gain-minus12 | single2 | 4.795 | 1.857 | 14.667 |
| vowels5 | gain-minus12 | single3 | 4.110 | 1.000 | 14.667 |
| vowels5 | gain-minus12 | single4 | 3.490 | 0.143 | 18.667 |
| vowels5 | gain-minus12 | single5 | 4.762 | 0.000 | 32.667 |
| vowels5 | gain-minus12 | drop1 | 4.468 | 2.381 | 10.667 |
| vowels5 | gain-minus12 | drop2 | 4.110 | 1.175 | 9.333 |
| vowels5 | gain-minus12 | drop3 | 4.289 | 1.492 | 10.889 |
| vowels5 | gain-minus12 | drop4 | 3.343 | 0.222 | 14.222 |
| vowels5 | gain-minus12 | drop5 | 4.077 | 1.714 | 8.667 |
| vowels4 | gain-minus12 | original | 4.667 | 0.000 | 27.333 |
| vowels4 | gain-minus12 | clean | 4.413 | 0.000 | 34.667 |
| vowels4 | gain-minus12 | augmented | 4.000 | 1.698 | 6.000 |
| vowels4 | gain-minus12 | single1 | 2.000 | 0.048 | 7.111 |
| vowels4 | gain-minus12 | single2 | 4.667 | 1.905 | 12.000 |
| vowels4 | gain-minus12 | single3 | 4.000 | 1.048 | 12.000 |
| vowels4 | gain-minus12 | single4 | 3.571 | 0.143 | 16.000 |
| vowels4 | gain-minus12 | single5 | 4.667 | 0.000 | 31.333 |
| vowels4 | gain-minus12 | drop1 | 4.444 | 2.429 | 8.000 |
| vowels4 | gain-minus12 | drop2 | 4.000 | 1.175 | 6.667 |
| vowels4 | gain-minus12 | drop3 | 4.270 | 1.492 | 8.222 |
| vowels4 | gain-minus12 | drop4 | 3.286 | 0.222 | 11.556 |
| vowels4 | gain-minus12 | drop5 | 4.000 | 1.714 | 6.000 |
| vowels5 | white-snr20 | original | 13.699 | 11.667 | 18.000 |
| vowels5 | white-snr20 | clean | 13.487 | 8.857 | 20.444 |
| vowels5 | white-snr20 | augmented | 4.795 | 6.397 | 6.444 |
| vowels5 | white-snr20 | single1 | 13.633 | 15.063 | 12.000 |
| vowels5 | white-snr20 | single2 | 7.469 | 1.810 | 25.333 |
| vowels5 | white-snr20 | single3 | 6.849 | 11.190 | 3.333 |
| vowels5 | white-snr20 | single4 | 15.460 | 10.254 | 26.889 |
| vowels5 | white-snr20 | single5 | 12.639 | 9.921 | 17.333 |
| vowels5 | white-snr20 | drop1 | 5.316 | 6.492 | 6.889 |
| vowels5 | white-snr20 | drop2 | 5.577 | 9.302 | 3.333 |
| vowels5 | white-snr20 | drop3 | 6.833 | 1.937 | 19.111 |
| vowels5 | white-snr20 | drop4 | 5.577 | 6.476 | 8.000 |
| vowels5 | white-snr20 | drop5 | 4.795 | 6.397 | 6.667 |
| vowels4 | white-snr20 | original | 13.333 | 12.095 | 15.333 |
| vowels4 | white-snr20 | clean | 13.333 | 9.238 | 17.778 |
| vowels4 | white-snr20 | augmented | 4.698 | 6.571 | 3.778 |
| vowels4 | white-snr20 | single1 | 13.349 | 15.508 | 9.333 |
| vowels4 | white-snr20 | single2 | 7.333 | 1.810 | 23.333 |
| vowels4 | white-snr20 | single3 | 6.667 | 11.429 | 0.667 |
| vowels4 | white-snr20 | single4 | 15.365 | 10.635 | 25.556 |
| vowels4 | white-snr20 | single5 | 12.587 | 10.349 | 14.667 |
| vowels4 | white-snr20 | drop1 | 5.238 | 6.651 | 4.222 |
| vowels4 | white-snr20 | drop2 | 5.540 | 9.540 | 0.667 |
| vowels4 | white-snr20 | drop3 | 6.746 | 1.952 | 17.111 |
| vowels4 | white-snr20 | drop4 | 5.556 | 6.714 | 5.333 |
| vowels4 | white-snr20 | drop5 | 5.127 | 6.540 | 4.000 |
| vowels5 | white-snr10 | original | 30.137 | 3.333 | 80.667 |
| vowels5 | white-snr10 | clean | 30.137 | 1.794 | 87.556 |
| vowels5 | white-snr10 | augmented | 13.699 | 1.698 | 63.111 |
| vowels5 | white-snr10 | single1 | 25.734 | 13.778 | 49.333 |
| vowels5 | white-snr10 | single2 | 21.233 | 0.000 | 99.333 |
| vowels5 | white-snr10 | single3 | 14.693 | 7.190 | 30.000 |
| vowels5 | white-snr10 | single4 | 35.535 | 1.206 | 99.556 |
| vowels5 | white-snr10 | single5 | 29.452 | 3.857 | 76.667 |
| vowels5 | white-snr10 | drop1 | 13.324 | 1.286 | 66.222 |
| vowels5 | white-snr10 | drop2 | 14.840 | 3.730 | 55.778 |
| vowels5 | white-snr10 | drop3 | 22.619 | 0.000 | 99.333 |
| vowels5 | white-snr10 | drop4 | 13.421 | 1.444 | 64.222 |
| vowels5 | white-snr10 | drop5 | 13.699 | 1.571 | 63.333 |
| vowels4 | white-snr10 | original | 30.095 | 3.571 | 80.000 |
| vowels4 | white-snr10 | clean | 30.000 | 2.032 | 87.556 |
| vowels4 | white-snr10 | augmented | 13.952 | 1.698 | 61.111 |
| vowels4 | white-snr10 | single1 | 25.762 | 14.270 | 47.778 |
| vowels4 | white-snr10 | single2 | 20.667 | 0.000 | 99.333 |
| vowels4 | white-snr10 | single3 | 14.635 | 7.381 | 27.333 |
| vowels4 | white-snr10 | single4 | 35.302 | 1.302 | 99.556 |
| vowels4 | white-snr10 | single5 | 29.333 | 4.190 | 75.333 |
| vowels4 | white-snr10 | drop1 | 13.143 | 1.302 | 64.222 |
| vowels4 | white-snr10 | drop2 | 15.016 | 3.889 | 53.778 |
| vowels4 | white-snr10 | drop3 | 22.698 | 0.000 | 99.333 |
| vowels4 | white-snr10 | drop4 | 13.524 | 1.444 | 62.222 |
| vowels4 | white-snr10 | drop5 | 14.000 | 1.571 | 61.333 |
| vowels5 | band-300-3400 | original | 36.301 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | clean | 35.127 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | augmented | 26.484 | 0.000 | 97.333 |
| vowels5 | band-300-3400 | single1 | 34.035 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | single2 | 34.883 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | single3 | 36.040 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | single4 | 9.736 | 2.254 | 39.111 |
| vowels5 | band-300-3400 | single5 | 35.013 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | drop1 | 26.419 | 0.048 | 97.333 |
| vowels5 | band-300-3400 | drop2 | 24.658 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | drop3 | 24.543 | 0.016 | 97.333 |
| vowels5 | band-300-3400 | drop4 | 36.448 | 0.000 | 100.000 |
| vowels5 | band-300-3400 | drop5 | 26.517 | 0.000 | 97.333 |
| vowels4 | band-300-3400 | original | 36.667 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | clean | 35.381 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | augmented | 26.429 | 0.000 | 97.333 |
| vowels4 | band-300-3400 | single1 | 34.143 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | single2 | 35.016 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | single3 | 36.032 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | single4 | 9.619 | 2.254 | 37.111 |
| vowels4 | band-300-3400 | single5 | 35.333 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | drop1 | 26.349 | 0.048 | 97.333 |
| vowels4 | band-300-3400 | drop2 | 24.667 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | drop3 | 24.429 | 0.016 | 97.333 |
| vowels4 | band-300-3400 | drop4 | 36.508 | 0.000 | 100.000 |
| vowels4 | band-300-3400 | drop5 | 26.444 | 0.000 | 97.333 |
| vowels5 | reverb-rt60-0.3 | original | 2.642 | 0.476 | 7.333 |
| vowels5 | reverb-rt60-0.3 | clean | 2.055 | 0.143 | 8.667 |
| vowels5 | reverb-rt60-0.3 | augmented | 4.061 | 0.206 | 16.000 |
| vowels5 | reverb-rt60-0.3 | single1 | 2.055 | 0.302 | 8.000 |
| vowels5 | reverb-rt60-0.3 | single2 | 4.110 | 0.365 | 23.556 |
| vowels5 | reverb-rt60-0.3 | single3 | 4.012 | 0.175 | 21.556 |
| vowels5 | reverb-rt60-0.3 | single4 | 1.598 | 0.095 | 9.333 |
| vowels5 | reverb-rt60-0.3 | single5 | 2.055 | 0.143 | 8.222 |
| vowels5 | reverb-rt60-0.3 | drop1 | 4.110 | 0.349 | 16.000 |
| vowels5 | reverb-rt60-0.3 | drop2 | 2.919 | 0.333 | 14.667 |
| vowels5 | reverb-rt60-0.3 | drop3 | 3.425 | 0.190 | 16.000 |
| vowels5 | reverb-rt60-0.3 | drop4 | 3.327 | 0.063 | 16.667 |
| vowels5 | reverb-rt60-0.3 | drop5 | 4.110 | 0.206 | 16.222 |
| vowels4 | reverb-rt60-0.3 | original | 2.571 | 0.476 | 5.333 |
| vowels4 | reverb-rt60-0.3 | clean | 2.000 | 0.143 | 6.667 |
| vowels4 | reverb-rt60-0.3 | augmented | 3.952 | 0.206 | 14.000 |
| vowels4 | reverb-rt60-0.3 | single1 | 2.000 | 0.302 | 6.000 |
| vowels4 | reverb-rt60-0.3 | single2 | 4.667 | 0.365 | 22.222 |
| vowels4 | reverb-rt60-0.3 | single3 | 4.048 | 0.175 | 19.556 |
| vowels4 | reverb-rt60-0.3 | single4 | 1.714 | 0.095 | 7.333 |
| vowels4 | reverb-rt60-0.3 | single5 | 2.000 | 0.143 | 6.222 |
| vowels4 | reverb-rt60-0.3 | drop1 | 4.635 | 0.349 | 14.000 |
| vowels4 | reverb-rt60-0.3 | drop2 | 2.841 | 0.333 | 12.667 |
| vowels4 | reverb-rt60-0.3 | drop3 | 3.333 | 0.190 | 14.667 |
| vowels4 | reverb-rt60-0.3 | drop4 | 3.270 | 0.063 | 14.667 |
| vowels4 | reverb-rt60-0.3 | drop5 | 4.206 | 0.206 | 14.222 |
| vowels5 | CV | original | 3.959 | 0.441 | 44.222 |
| vowels5 | CV | clean | 4.131 | 0.438 | 44.481 |
| vowels5 | CV | augmented | 4.131 | 0.421 | 46.000 |
| vowels5 | CV | single1 | 4.131 | 0.381 | 44.444 |
| vowels5 | CV | single2 | 4.131 | 0.413 | 46.037 |
| vowels5 | CV | single3 | 3.442 | 0.375 | 44.370 |
| vowels5 | CV | single4 | 4.507 | 0.254 | 46.222 |
| vowels5 | CV | single5 | 4.125 | 0.464 | 44.407 |
| vowels5 | CV | drop1 | 4.131 | 0.424 | 45.370 |
| vowels5 | CV | drop2 | 4.131 | 0.446 | 45.222 |
| vowels5 | CV | drop3 | 4.309 | 0.437 | 45.926 |
| vowels5 | CV | drop4 | 4.149 | 0.396 | 46.259 |
| vowels5 | CV | drop5 | 4.168 | 0.411 | 46.000 |
| vowels4 | CV | original | 4.396 | 0.640 | 23.444 |
| vowels4 | CV | clean | 4.577 | 0.632 | 23.889 |
| vowels4 | CV | augmented | 4.762 | 0.607 | 26.037 |
| vowels4 | CV | single1 | 4.749 | 0.577 | 24.222 |
| vowels4 | CV | single2 | 4.533 | 0.589 | 26.407 |
| vowels4 | CV | single3 | 3.917 | 0.556 | 23.926 |
| vowels4 | CV | single4 | 4.790 | 0.421 | 26.407 |
| vowels4 | CV | single5 | 4.518 | 0.656 | 23.926 |
| vowels4 | CV | drop1 | 4.533 | 0.618 | 25.074 |
| vowels4 | CV | drop2 | 4.551 | 0.654 | 25.111 |
| vowels4 | CV | drop3 | 4.723 | 0.630 | 25.852 |
| vowels4 | CV | drop4 | 4.422 | 0.566 | 26.778 |
| vowels4 | CV | drop5 | 4.762 | 0.599 | 26.074 |

## 事前固定の候補スクリーニング

| arm | 通過 | 不合格項目 |
| --- | --- | --- |
| augmented | False | vowels5/clean/far +0.524pp; vowels5/clean/frr +2.000pp; vowels5/reverb-rt60-0.3/frr +7.333pp; vowels5/CV/frr +1.519pp; vowels4/clean/far +0.524pp; vowels4/clean/frr +2.000pp; vowels4/reverb-rt60-0.3/frr +7.333pp; vowels4/CV/frr +2.148pp |
| single1 | False | vowels5/white-snr20/eer +0.147pp; vowels4/white-snr20/eer +0.016pp |
| single2 | False | vowels5/clean/far +0.587pp; vowels5/clean/frr +2.000pp; vowels5/reverb-rt60-0.3/frr +14.889pp; vowels5/CV/frr +1.556pp; vowels4/clean/far +0.587pp; vowels4/clean/frr +2.000pp; vowels4/reverb-rt60-0.3/frr +15.556pp; vowels4/CV/frr +2.519pp |
| single3 | False | vowels5/clean/far +0.492pp; vowels5/reverb-rt60-0.3/frr +12.889pp; vowels4/clean/far +0.492pp; vowels4/reverb-rt60-0.3/frr +12.889pp |
| single4 | False | vowels5/clean/far +0.540pp; vowels5/CV/frr +1.741pp; vowels5/white-snr20/eer +1.973pp; vowels5/white-snr10/eer +5.398pp; vowels4/clean/far +0.540pp; vowels4/CV/frr +2.519pp; vowels4/white-snr20/eer +2.032pp; vowels4/white-snr10/eer +5.302pp |
| single5 | True | — |
| drop1 | False | vowels5/clean/far +0.778pp; vowels5/clean/frr +2.000pp; vowels5/reverb-rt60-0.3/frr +7.333pp; vowels4/clean/far +0.778pp; vowels4/clean/frr +2.000pp; vowels4/reverb-rt60-0.3/frr +7.333pp; vowels4/CV/frr +1.185pp |
| drop2 | False | vowels5/clean/far +0.476pp; vowels5/clean/frr +1.111pp; vowels5/reverb-rt60-0.3/frr +6.000pp; vowels4/clean/far +0.476pp; vowels4/clean/frr +1.111pp; vowels4/reverb-rt60-0.3/frr +6.000pp; vowels4/CV/frr +1.222pp |
| drop3 | False | vowels5/clean/far +0.476pp; vowels5/clean/frr +1.556pp; vowels5/reverb-rt60-0.3/frr +7.333pp; vowels5/CV/frr +1.444pp; vowels4/clean/far +0.476pp; vowels4/clean/frr +1.556pp; vowels4/reverb-rt60-0.3/frr +8.000pp; vowels4/CV/frr +1.963pp |
| drop4 | False | vowels5/clean/frr +2.000pp; vowels5/reverb-rt60-0.3/frr +8.000pp; vowels5/CV/frr +1.778pp; vowels4/clean/frr +2.000pp; vowels4/reverb-rt60-0.3/frr +8.000pp; vowels4/CV/frr +2.889pp |
| drop5 | False | vowels5/clean/far +0.540pp; vowels5/clean/frr +2.000pp; vowels5/reverb-rt60-0.3/frr +7.556pp; vowels5/CV/frr +1.519pp; vowels4/clean/far +0.540pp; vowels4/clean/frr +2.000pp; vowels4/reverb-rt60-0.3/frr +7.556pp; vowels4/CV/frr +2.185pp |

float64スコア再計算 2,997,000件（最大差 8.9e-08）、重み付きROCの別計算 2,220件、元PCM特徴36件とpair列・加工枠の整合性を検証。

全seed・対応付き信頼区間・校正閾値は[evaluation-results.json](evaluation-results.json)、設計と制約は[README](README.md)を参照。単独では加工率50%、除去では残り4条件を元の約10%ずつ保持する。区間は多重比較未補正で、採用の確証ではない。
