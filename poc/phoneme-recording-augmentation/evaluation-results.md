# encoder録音条件拡張：追加学習の対照比較

3 seedの指標の平均であり、ensembleではない。各モデルのclean validation FAR1%閾値を固定適用。観測済みtestでの探索。単位%。

## 3 seed平均

| 条件 | 受付 | 元EER | clean追加EER | 拡張追加EER | 元FAR | clean追加FAR | 拡張追加FAR | 元FRR | clean追加FRR | 拡張追加FRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | vowels5 | 1.370 | 0.962 | 1.631 | 0.333 | 0.190 | 0.714 | 4.667 | 4.667 | 6.667 |
| clean | vowels4 | 1.333 | 0.937 | 1.587 | 0.333 | 0.190 | 0.714 | 2.000 | 2.000 | 4.000 |
| gain-minus12 | vowels5 | 4.795 | 4.387 | 4.093 | 0.000 | 0.000 | 1.698 | 28.667 | 36.000 | 8.667 |
| gain-minus12 | vowels4 | 4.667 | 4.413 | 4.000 | 0.000 | 0.000 | 1.698 | 27.333 | 34.667 | 6.000 |
| white-snr20 | vowels5 | 13.699 | 13.487 | 4.795 | 11.667 | 8.857 | 6.397 | 18.000 | 20.444 | 6.444 |
| white-snr20 | vowels4 | 13.333 | 13.333 | 4.698 | 12.095 | 9.238 | 6.571 | 15.333 | 17.778 | 3.778 |
| white-snr10 | vowels5 | 30.137 | 30.137 | 13.699 | 3.333 | 1.794 | 1.698 | 80.667 | 87.556 | 63.111 |
| white-snr10 | vowels4 | 30.095 | 30.000 | 13.952 | 3.571 | 2.032 | 1.698 | 80.000 | 87.556 | 61.111 |
| band-300-3400 | vowels5 | 36.301 | 35.127 | 26.484 | 0.000 | 0.000 | 0.000 | 100.000 | 100.000 | 97.333 |
| band-300-3400 | vowels4 | 36.667 | 35.381 | 26.429 | 0.000 | 0.000 | 0.000 | 100.000 | 100.000 | 97.333 |
| reverb-rt60-0.3 | vowels5 | 2.642 | 2.055 | 4.061 | 0.476 | 0.143 | 0.206 | 7.333 | 8.667 | 16.000 |
| reverb-rt60-0.3 | vowels4 | 2.571 | 2.000 | 3.952 | 0.476 | 0.143 | 0.206 | 5.333 | 6.667 | 14.000 |
| CV | vowels5 | 3.959 | 4.131 | 4.131 | 0.441 | 0.438 | 0.421 | 44.222 | 44.481 | 46.000 |
| CV | vowels4 | 4.396 | 4.577 | 4.762 | 0.640 | 0.632 | 0.607 | 23.444 | 23.889 | 26.037 |

## 全seedの結果

| 条件 | 受付 | モデル | EER | FAR | FRR |
| --- | --- | --- | ---: | ---: | ---: |
| clean | vowels5 | original | 1.370 | 0.333 | 4.667 |
| clean | vowels4 | original | 1.333 | 0.333 | 2.000 |
| gain-minus12 | vowels5 | original | 4.795 | 0.000 | 28.667 |
| gain-minus12 | vowels4 | original | 4.667 | 0.000 | 27.333 |
| white-snr20 | vowels5 | original | 13.699 | 11.667 | 18.000 |
| white-snr20 | vowels4 | original | 13.333 | 12.095 | 15.333 |
| white-snr10 | vowels5 | original | 30.137 | 3.333 | 80.667 |
| white-snr10 | vowels4 | original | 30.095 | 3.571 | 80.000 |
| band-300-3400 | vowels5 | original | 36.301 | 0.000 | 100.000 |
| band-300-3400 | vowels4 | original | 36.667 | 0.000 | 100.000 |
| reverb-rt60-0.3 | vowels5 | original | 2.642 | 0.476 | 7.333 |
| reverb-rt60-0.3 | vowels4 | original | 2.571 | 0.476 | 5.333 |
| clean | vowels5 | clean-20261028 | 0.978 | 0.190 | 4.667 |
| clean | vowels4 | clean-20261028 | 0.952 | 0.190 | 2.000 |
| gain-minus12 | vowels5 | clean-20261028 | 4.354 | 0.000 | 36.000 |
| gain-minus12 | vowels4 | clean-20261028 | 4.381 | 0.000 | 34.667 |
| white-snr20 | vowels5 | clean-20261028 | 13.552 | 8.810 | 20.000 |
| white-snr20 | vowels4 | clean-20261028 | 13.333 | 9.190 | 17.333 |
| white-snr10 | vowels5 | clean-20261028 | 30.137 | 1.762 | 87.333 |
| white-snr10 | vowels4 | clean-20261028 | 30.000 | 2.000 | 87.333 |
| band-300-3400 | vowels5 | clean-20261028 | 35.029 | 0.000 | 100.000 |
| band-300-3400 | vowels4 | clean-20261028 | 35.333 | 0.000 | 100.000 |
| reverb-rt60-0.3 | vowels5 | clean-20261028 | 2.055 | 0.143 | 8.667 |
| reverb-rt60-0.3 | vowels4 | clean-20261028 | 2.000 | 0.143 | 6.667 |
| clean | vowels5 | clean-20261029 | 0.930 | 0.190 | 4.667 |
| clean | vowels4 | clean-20261029 | 0.905 | 0.190 | 2.000 |
| gain-minus12 | vowels5 | clean-20261029 | 4.403 | 0.000 | 36.000 |
| gain-minus12 | vowels4 | clean-20261029 | 4.429 | 0.000 | 34.667 |
| white-snr20 | vowels5 | clean-20261029 | 13.356 | 9.048 | 20.667 |
| white-snr20 | vowels4 | clean-20261029 | 13.333 | 9.429 | 18.000 |
| white-snr10 | vowels5 | clean-20261029 | 30.137 | 1.952 | 86.000 |
| white-snr10 | vowels4 | clean-20261029 | 30.000 | 2.190 | 86.000 |
| band-300-3400 | vowels5 | clean-20261029 | 34.932 | 0.000 | 100.000 |
| band-300-3400 | vowels4 | clean-20261029 | 35.333 | 0.000 | 100.000 |
| reverb-rt60-0.3 | vowels5 | clean-20261029 | 2.055 | 0.143 | 8.667 |
| reverb-rt60-0.3 | vowels4 | clean-20261029 | 2.000 | 0.143 | 6.667 |
| clean | vowels5 | clean-20261030 | 0.978 | 0.190 | 4.667 |
| clean | vowels4 | clean-20261030 | 0.952 | 0.190 | 2.000 |
| gain-minus12 | vowels5 | clean-20261030 | 4.403 | 0.000 | 36.000 |
| gain-minus12 | vowels4 | clean-20261030 | 4.429 | 0.000 | 34.667 |
| white-snr20 | vowels5 | clean-20261030 | 13.552 | 8.714 | 20.667 |
| white-snr20 | vowels4 | clean-20261030 | 13.333 | 9.095 | 18.000 |
| white-snr10 | vowels5 | clean-20261030 | 30.137 | 1.667 | 89.333 |
| white-snr10 | vowels4 | clean-20261030 | 30.000 | 1.905 | 89.333 |
| band-300-3400 | vowels5 | clean-20261030 | 35.421 | 0.000 | 100.000 |
| band-300-3400 | vowels4 | clean-20261030 | 35.476 | 0.000 | 100.000 |
| reverb-rt60-0.3 | vowels5 | clean-20261030 | 2.055 | 0.143 | 8.667 |
| reverb-rt60-0.3 | vowels4 | clean-20261030 | 2.000 | 0.143 | 6.667 |
| clean | vowels5 | augmented-20261028 | 1.566 | 0.714 | 6.667 |
| clean | vowels4 | augmented-20261028 | 1.524 | 0.714 | 4.000 |
| gain-minus12 | vowels5 | augmented-20261028 | 4.110 | 1.667 | 8.667 |
| gain-minus12 | vowels4 | augmented-20261028 | 4.000 | 1.667 | 6.000 |
| white-snr20 | vowels5 | augmented-20261028 | 4.795 | 6.429 | 6.667 |
| white-snr20 | vowels4 | augmented-20261028 | 4.714 | 6.619 | 4.000 |
| white-snr10 | vowels5 | augmented-20261028 | 13.699 | 1.667 | 63.333 |
| white-snr10 | vowels4 | augmented-20261028 | 14.000 | 1.667 | 61.333 |
| band-300-3400 | vowels5 | augmented-20261028 | 26.712 | 0.000 | 97.333 |
| band-300-3400 | vowels4 | augmented-20261028 | 26.667 | 0.000 | 97.333 |
| reverb-rt60-0.3 | vowels5 | augmented-20261028 | 4.110 | 0.190 | 16.000 |
| reverb-rt60-0.3 | vowels4 | augmented-20261028 | 4.000 | 0.190 | 14.000 |
| clean | vowels5 | augmented-20261029 | 1.614 | 0.714 | 6.667 |
| clean | vowels4 | augmented-20261029 | 1.571 | 0.714 | 4.000 |
| gain-minus12 | vowels5 | augmented-20261029 | 4.110 | 1.619 | 8.667 |
| gain-minus12 | vowels4 | augmented-20261029 | 4.000 | 1.619 | 6.000 |
| white-snr20 | vowels5 | augmented-20261029 | 4.795 | 6.429 | 6.000 |
| white-snr20 | vowels4 | augmented-20261029 | 4.714 | 6.619 | 3.333 |
| white-snr10 | vowels5 | augmented-20261029 | 13.699 | 1.810 | 62.667 |
| white-snr10 | vowels4 | augmented-20261029 | 13.857 | 1.810 | 60.667 |
| band-300-3400 | vowels5 | augmented-20261029 | 26.712 | 0.000 | 97.333 |
| band-300-3400 | vowels4 | augmented-20261029 | 26.667 | 0.000 | 97.333 |
| reverb-rt60-0.3 | vowels5 | augmented-20261029 | 3.963 | 0.190 | 16.000 |
| reverb-rt60-0.3 | vowels4 | augmented-20261029 | 3.857 | 0.190 | 14.000 |
| clean | vowels5 | augmented-20261030 | 1.712 | 0.714 | 6.667 |
| clean | vowels4 | augmented-20261030 | 1.667 | 0.714 | 4.000 |
| gain-minus12 | vowels5 | augmented-20261030 | 4.061 | 1.810 | 8.667 |
| gain-minus12 | vowels4 | augmented-20261030 | 4.000 | 1.810 | 6.000 |
| white-snr20 | vowels5 | augmented-20261030 | 4.795 | 6.333 | 6.667 |
| white-snr20 | vowels4 | augmented-20261030 | 4.667 | 6.476 | 4.000 |
| white-snr10 | vowels5 | augmented-20261030 | 13.699 | 1.619 | 63.333 |
| white-snr10 | vowels4 | augmented-20261030 | 14.000 | 1.619 | 61.333 |
| band-300-3400 | vowels5 | augmented-20261030 | 26.027 | 0.000 | 97.333 |
| band-300-3400 | vowels4 | augmented-20261030 | 25.952 | 0.000 | 97.333 |
| reverb-rt60-0.3 | vowels5 | augmented-20261030 | 4.110 | 0.238 | 16.000 |
| reverb-rt60-0.3 | vowels4 | augmented-20261030 | 4.000 | 0.238 | 14.000 |
| CV | vowels5 | original | 3.959 | 0.441 | 44.222 |
| CV | vowels4 | original | 4.396 | 0.640 | 23.444 |
| CV | vowels5 | clean-20261028 | 4.131 | 0.433 | 44.333 |
| CV | vowels4 | clean-20261028 | 4.518 | 0.628 | 23.667 |
| CV | vowels5 | clean-20261029 | 4.131 | 0.425 | 44.667 |
| CV | vowels4 | clean-20261029 | 4.627 | 0.625 | 24.000 |
| CV | vowels5 | clean-20261030 | 4.131 | 0.456 | 44.444 |
| CV | vowels4 | clean-20261030 | 4.585 | 0.644 | 24.000 |
| CV | vowels5 | augmented-20261028 | 4.131 | 0.414 | 46.000 |
| CV | vowels4 | augmented-20261028 | 4.762 | 0.594 | 25.889 |
| CV | vowels5 | augmented-20261029 | 4.131 | 0.421 | 46.000 |
| CV | vowels4 | augmented-20261029 | 4.762 | 0.609 | 26.111 |
| CV | vowels5 | augmented-20261030 | 4.131 | 0.429 | 46.000 |
| CV | vowels4 | augmented-20261030 | 4.762 | 0.617 | 26.111 |

## 検証・適用範囲

学習は既存140話者・31,671区間・35音素。今回は /dy/ のpositive pairを確保できず追加学習対象外。元encoderの構造・評価支持集合は維持。各条件/seedは6,000更新。

float64スコア再計算 567,000件、最大差 8.68e-08。元モデル再現 81,000件、train元PCM特徴再計算 36件、全seedのpair列と初期checkpoint一致を確認。

[全数値・対応付きbootstrap・閾値](evaluation-results.json)。学習/評価音声は分離。ただし拡張条件を決める前に評価話者の結果を参照しているため、未使用testでの再現確認は別途必要。各seedの指標を平均した区間は固定した3モデルの話者再抽出によるもので、学習seed一般の不確実性を保証しない。元境界固定の診断で、強い雑音時のalignment失敗を含むend-to-end評価ではない。
