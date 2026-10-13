# 34音素に合わせたTransformer再学習

36→36＝36音素学習・36音素照合、36→34＝照合だけch/k除外、34→34＝学習・照合ともch/k除外。固定encoderの3音響seed平均。

## EER

| 集合 | checkpoint | コーパス | 36→36 | 36→34 | 34→34 | 再学習−照合時除外 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| validation | selected | jvs | 0.411% | 0.415% | 0.437% | +0.023 pp |
| validation | selected | common_voice | 3.199% | 3.338% | 3.191% | -0.147 pp |
| validation | selected | jtube | 0.835% | 0.899% | 0.987% | +0.088 pp |
| validation | selected | macro | 1.482% | 1.551% | 1.538% | -0.012 pp |
| validation | final | jvs | 0.635% | 0.726% | 0.606% | -0.120 pp |
| validation | final | common_voice | 4.643% | 4.692% | 3.833% | -0.860 pp |
| validation | final | jtube | 1.139% | 1.139% | 1.215% | +0.076 pp |
| validation | final | macro | 2.139% | 2.186% | 1.884% | -0.301 pp |
| test | selected | jvs | 1.001% | 0.926% | 0.748% | -0.178 pp |
| test | selected | common_voice | 2.556% | 2.400% | 2.993% | +0.594 pp |
| test | selected | jtube | 0.929% | 0.938% | 1.041% | +0.103 pp |
| test | selected | macro | 1.495% | 1.421% | 1.594% | +0.173 pp |
| test | final | jvs | 1.186% | 1.079% | 1.105% | +0.026 pp |
| test | final | common_voice | 2.483% | 2.499% | 2.639% | +0.140 pp |
| test | final | jtube | 1.197% | 1.315% | 1.114% | -0.200 pp |
| test | final | macro | 1.622% | 1.631% | 1.619% | -0.011 pp |

## validationで選んだtestモデルの固定閾値性能

| コーパス | 条件 | FAR | 全入力FRR | 採点率 |
| --- | --- | ---: | ---: | ---: |
| jvs | 36→36 | 0.724% | 3.333% | 98.000% |
| jvs | 36→34 | 0.724% | 3.200% | 98.000% |
| jvs | 34→34 | 0.835% | 2.667% | 98.000% |
| common_voice | 36→36 | 0.852% | 41.370% | 64.556% |
| common_voice | 36→34 | 0.895% | 40.222% | 64.556% |
| common_voice | 34→34 | 0.854% | 42.037% | 64.556% |
| jtube | 36→36 | 1.035% | 40.667% | 59.833% |
| jtube | 36→34 | 1.006% | 40.667% | 59.833% |
| jtube | 34→34 | 0.968% | 40.611% | 59.833% |
| macro | 36→36 | 0.870% | 28.457% | 74.130% |
| macro | 36→34 | 0.875% | 28.030% | 74.130% |
| macro | 34→34 | 0.886% | 28.438% | 74.130% |

## 再学習のEER差と対応付き95%区間

| checkpoint | コーパス | 比較対象 | 差 | 改善 / 同値 / 悪化 seed | 95%区間 |
| --- | --- | --- | ---: | --- | --- |
| selected | jvs | train36_score36 | -0.253 pp | 3 / 0 / 0 | [-0.426, -0.003] pp |
| selected | jvs | train36_score34 | -0.178 pp | 3 / 0 / 0 | [-0.312, +0.061] pp |
| selected | common_voice | train36_score36 | +0.437 pp | 1 / 0 / 2 | [+0.055, +0.691] pp |
| selected | common_voice | train36_score34 | +0.594 pp | 1 / 0 / 2 | [+0.079, +0.908] pp |
| selected | jtube | train36_score36 | +0.112 pp | 1 / 0 / 2 | [-0.157, +0.290] pp |
| selected | jtube | train36_score34 | +0.103 pp | 1 / 0 / 2 | [-0.195, +0.274] pp |
| selected | macro | train36_score36 | +0.099 pp | 1 / 0 / 2 | [-0.078, +0.226] pp |
| selected | macro | train36_score34 | +0.173 pp | 1 / 0 / 2 | [-0.032, +0.312] pp |
| final | jvs | train36_score36 | -0.081 pp | 2 / 0 / 1 | [-0.439, +0.235] pp |
| final | jvs | train36_score34 | +0.026 pp | 1 / 0 / 2 | [-0.275, +0.322] pp |
| final | common_voice | train36_score36 | +0.156 pp | 1 / 0 / 2 | [-0.265, +1.040] pp |
| final | common_voice | train36_score34 | +0.140 pp | 1 / 0 / 2 | [-0.298, +0.870] pp |
| final | jtube | train36_score36 | -0.083 pp | 1 / 1 / 1 | [-0.429, +0.060] pp |
| final | jtube | train36_score34 | -0.200 pp | 2 / 0 / 1 | [-0.512, +0.000] pp |
| final | macro | train36_score36 | -0.003 pp | 2 / 0 / 1 | [-0.221, +0.317] pp |
| final | macro | train36_score34 | -0.011 pp | 2 / 0 / 1 | [-0.228, +0.265] pp |

## 選択更新

| 音響seed | 36音素学習 | 34音素学習 |
| --- | ---: | ---: |
| 20260926 | 58000 | 58000 |
| 20260927 | 22000 | 50000 |
| 20260928 | 49000 | 50000 |

学習ペア・初期値・更新数・LR・encoderは固定。モデルとvalidation閾値を凍結してからtest推論。selectedでは各学習条件のvalidation最良モデルを使うため、学習とcheckpoint選択の効果を含む。36→34の対照は元の36音素validationで選ばれたcheckpointを維持し、34音素向けに選び直していない。finalは全条件64k固定。

34音素は前実験で固定された候補で、今回は再探索していない。全testは既観測。前回の探索/確認validation話者も今回はcheckpoint選択に使うため、独立確認とは扱わない。CIは固定モデル・固定音素集合を条件とした対応付き話者bootstrapで、学習・選択の変動を含まない。EERは採点可能分、FAR/FRRは未採点を含む。

実行成果物: `artifacts/phoneme-subset-retraining/src490-34phones-20261014-v1`。条件・再現方法は[README](README.md)。全seed・JVS別テキスト・監査は[evaluation-results.json](evaluation-results.json)。
