# 固定SRC490モデルの音素選別

選択: a / i / u / e / o / N / b / by / cl / d / dy / f / g / gy / h / hy / j / ky / m / my / n / ny / p / py / r / ry / s / sh / t / ts / v / w / y / z（34種類）。
除外: ch / k。探索した異なる集合は140通り。

3音響seed平均。EERは採点可能分、FAR/FRRは未採点を拒否に含む全入力。平均差はselected−all36、単位はパーセントポイント。

| 集合 | コーパス | 全36 EER | 選択後 EER | 差 | 改善 / 悪化 seed |
| --- | --- | ---: | ---: | ---: | --- |
| validation/discovery | jvs | 0.462% | 0.401% | -0.061 | 2 / 0 |
| validation/discovery | common_voice | 2.835% | 2.835% | +0.000 | 0 / 0 |
| validation/discovery | jtube | 0.703% | 0.617% | -0.086 | 2 / 1 |
| validation/discovery | macro | 1.333% | 1.284% | -0.049 | 2 / 1 |
| validation/confirmation | jvs | 0.437% | 0.340% | -0.097 | 2 / 0 |
| validation/confirmation | common_voice | 3.494% | 3.748% | +0.254 | 0 / 2 |
| validation/confirmation | jtube | 1.495% | 1.345% | -0.149 | 1 / 0 |
| validation/confirmation | macro | 1.809% | 1.811% | +0.002 | 2 / 1 |
| validation/full | jvs | 0.411% | 0.415% | +0.003 | 2 / 1 |
| validation/full | common_voice | 3.199% | 3.338% | +0.139 | 0 / 2 |
| validation/full | jtube | 0.835% | 0.899% | +0.064 | 1 / 2 |
| validation/full | macro | 1.482% | 1.551% | +0.069 | 1 / 2 |
| test/full | jvs | 1.001% | 0.926% | -0.075 | 3 / 0 |
| test/full | common_voice | 2.556% | 2.400% | -0.156 | 2 / 1 |
| test/full | jtube | 0.929% | 0.938% | +0.010 | 0 / 1 |
| test/full | macro | 1.495% | 1.421% | -0.074 | 3 / 0 |

## 固定validation FAR1%閾値でのtest

| コーパス | FAR 全36→選択 | FRR 全36→選択 | 採点率 | EER差の対応付き95%区間 |
| --- | ---: | ---: | ---: | --- |
| jvs | 0.724→0.724% | 3.333→3.200% | 98.000% | [-0.256, +0.052] pp |
| common_voice | 0.852→0.895% | 41.370→40.222% | 64.556% | [-0.349, +0.053] pp |
| jtube | 1.035→1.006% | 40.667→40.667% | 59.833% | [-0.020, +0.109] pp |
| macro | 0.870→0.875% | 28.457→28.030% | 74.130% | [-0.152, +0.017] pp |

## 採用までの探索

| 操作 | 除外 | 追加 | 音素数 | 探索EER |
| --- | --- | --- | ---: | ---: |
| start |  |  | 36 | 1.333% |
| delete | k |  | 35 | 1.298% |
| delete | ch |  | 34 | 1.284% |

## seed別test EER

| コーパス | 音響seed | 全36 | 選択後 |
| --- | ---: | ---: | ---: |
| jvs | 20260926 | 1.137% | 1.088% |
| jvs | 20260927 | 0.914% | 0.875% |
| jvs | 20260928 | 0.952% | 0.816% |
| common_voice | 20260926 | 2.778% | 2.789% |
| common_voice | 20260927 | 2.439% | 2.344% |
| common_voice | 20260928 | 2.451% | 2.065% |
| jtube | 20260926 | 1.114% | 1.114% |
| jtube | 20260927 | 0.836% | 0.865% |
| jtube | 20260928 | 0.836% | 0.836% |

## 解釈の範囲

探索話者と確認話者はquery・claimed speakerの両方で分離した。確認用話者は音素探索には未使用だが、元モデルのcheckpoint選択には使用済みであり、完全に独立したholdoutではない。確認結果で選択を変更せず、test前に集合・全validation閾値を固定した。

testも既観測の同じ話者集合。3音響seedは新規母集団ではない。95%区間は固定モデル・固定集合を条件とし、探索や学習の変動を含まない。母音も除外対象だが、少なくとも1母音を残し、全条件で従来の5母音必須ゲートを維持した。採点率改善は今回の対象外。

Transformerのattention・softmaxを毎回再計算した。保存済み重みの再正規化だけではない。選別で実使用音声量も減るため、固定総時間の比較ではない。探索は上限付きの局所探索で、最適集合や音素固有の普遍的な優劣を確定しない。再学習は未実施。

成果物: `artifacts/phoneme-subset-selection/src490-20261013-v2`。再現条件は[README](README.md)と[config.json](config.json)。
