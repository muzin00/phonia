# 実測EERを報酬にしたTransformerの強化学習

SRC490・36音素、固定encoderの3音響seed平均。BCEは対照、weightは重み付け方策、subsetは音素選択方策。
各条件512発話/1,024照合batch・500更新。RLは各batchで4回行動を抽選。selectedはvalidation選定、finalは固定終点。

## EER

| 集合 | コーパス | 追加前 | BCE選定 | 重みRL選定 | 選択RL選定 | BCE最終 | 重みRL最終 | 選択RL最終 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | jvs | 0.411% | 0.411% | 0.392% | 0.411% | 0.544% | 0.441% | 0.619% |
| validation | common_voice | 3.199% | 3.199% | 3.191% | 3.199% | 3.953% | 3.289% | 3.604% |
| validation | jtube | 0.835% | 0.835% | 0.835% | 0.835% | 1.007% | 0.911% | 0.995% |
| validation | macro | 1.482% | 1.482% | 1.473% | 1.482% | 1.835% | 1.547% | 1.739% |
| test | jvs | 1.001% | 1.001% | 1.033% | 1.001% | 1.231% | 1.027% | 1.127% |
| test | common_voice | 2.556% | 2.556% | 2.461% | 2.556% | 2.267% | 2.313% | 2.609% |
| test | jtube | 0.929% | 0.929% | 0.929% | 0.929% | 1.236% | 0.929% | 1.315% |
| test | macro | 1.495% | 1.495% | 1.474% | 1.495% | 1.578% | 1.423% | 1.684% |

## RL − BCEのtest EER差

| コーパス | 方策 | checkpoint | 差 | 改善 / 同値 / 悪化seed | 対応付き95%区間 |
| --- | --- | --- | ---: | --- | --- |
| jvs | weight | selected | +0.032 pp | 0 / 1 / 2 | [-0.021, +0.092] pp |
| jvs | weight | final | -0.204 pp | 3 / 0 / 0 | [-0.444, +0.021] pp |
| jvs | subset | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| jvs | subset | final | -0.104 pp | 2 / 0 / 1 | [-0.259, +0.148] pp |
| common_voice | weight | selected | -0.095 pp | 2 / 1 / 0 | [-0.167, +0.124] pp |
| common_voice | weight | final | +0.046 pp | 1 / 0 / 2 | [-0.365, +0.718] pp |
| common_voice | subset | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| common_voice | subset | final | +0.342 pp | 0 / 0 / 3 | [-0.275, +1.169] pp |
| jtube | weight | selected | +0.000 pp | 0 / 3 / 0 | [-0.019, +0.061] pp |
| jtube | weight | final | -0.308 pp | 3 / 0 / 0 | [-0.731, +0.092] pp |
| jtube | subset | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| jtube | subset | final | +0.078 pp | 1 / 0 / 2 | [-0.263, +0.514] pp |
| macro | weight | selected | -0.021 pp | 2 / 1 / 0 | [-0.049, +0.056] pp |
| macro | weight | final | -0.155 pp | 3 / 0 / 0 | [-0.370, +0.127] pp |
| macro | subset | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| macro | subset | final | +0.106 pp | 1 / 0 / 2 | [-0.129, +0.425] pp |

## validation FAR1%閾値を固定したtest性能

| コーパス | 条件 | FAR | 全入力FRR | 採点率 |
| --- | --- | ---: | ---: | ---: |
| jvs | initial | 0.724% | 3.333% | 98.000% |
| jvs | bce-selected | 0.724% | 3.333% | 98.000% |
| jvs | weight-selected | 0.717% | 3.467% | 98.000% |
| jvs | subset-selected | 0.724% | 3.333% | 98.000% |
| jvs | bce-final | 0.737% | 3.956% | 98.000% |
| jvs | weight-final | 0.717% | 3.511% | 98.000% |
| jvs | subset-final | 0.737% | 3.644% | 98.000% |
| common_voice | initial | 0.852% | 41.370% | 64.556% |
| common_voice | bce-selected | 0.852% | 41.370% | 64.556% |
| common_voice | weight-selected | 0.851% | 41.185% | 64.556% |
| common_voice | subset-selected | 0.852% | 41.370% | 64.556% |
| common_voice | bce-final | 0.939% | 38.370% | 64.556% |
| common_voice | weight-final | 0.860% | 40.667% | 64.556% |
| common_voice | subset-final | 0.880% | 47.296% | 64.556% |
| jtube | initial | 1.035% | 40.667% | 59.833% |
| jtube | bce-selected | 1.035% | 40.667% | 59.833% |
| jtube | weight-selected | 1.058% | 40.611% | 59.833% |
| jtube | subset-selected | 1.035% | 40.667% | 59.833% |
| jtube | bce-final | 1.146% | 40.722% | 59.833% |
| jtube | weight-final | 1.038% | 40.667% | 59.833% |
| jtube | subset-final | 1.158% | 40.833% | 59.833% |
| macro | initial | 0.870% | 28.457% | 74.130% |
| macro | bce-selected | 0.870% | 28.457% | 74.130% |
| macro | weight-selected | 0.876% | 28.421% | 74.130% |
| macro | subset-selected | 0.870% | 28.457% | 74.130% |
| macro | bce-final | 0.940% | 27.683% | 74.130% |
| macro | weight-final | 0.872% | 28.281% | 74.130% |
| macro | subset-final | 0.925% | 30.591% | 74.130% |

## validationで選択された追加更新数

| 音響seed | 出発点の旧更新 | BCE | 重みRL | 選択RL |
| --- | ---: | ---: | ---: | ---: |
| 20260926 | 58000 | 0 | 400 | 0 |
| 20260927 | 22000 | 0 | 400 | 0 |
| 20260928 | 49000 | 0 | 0 | 0 |

## 学習中の行動探索

| seed | 方策 | 平均報酬EER | rollout間EER標準偏差 | 非ゼロadvantage率 | 平均選択音素数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 20260926 | weight | 1.444% | 0.203 pp | 48.8% | 16.60 |
| 20260926 | subset | 2.616% | 0.661 pp | 86.4% | 8.29 |
| 20260927 | weight | 1.732% | 0.184 pp | 44.3% | 16.60 |
| 20260927 | subset | 2.857% | 0.667 pp | 86.3% | 8.51 |
| 20260928 | weight | 1.486% | 0.188 pp | 44.0% | 16.60 |
| 20260928 | subset | 2.634% | 0.644 pp | 85.5% | 8.79 |

報酬は学習batchの実測EER。正解ラベルは報酬計算だけに用い、方策の入力には含めない。探索中の報酬EERと決定的推論のEERは別の量。選択方策は音素数を直接減らす報酬を持たない。
EERは採点可能分、FAR/FRRは未採点を拒否として含む。採点対象は全条件固定。test FARは同値に揃えていない。
3音響seed・1方策seed・単一設定。全testは既観測。CIは固定モデルを条件とした話者bootstrapで学習・選択の変動を含まない。計算量と構造も条件間で異なる。標準モデルは置き換えない。

成果物: `artifacts/phoneme-eer-reinforcement/src490-reinforce-20261015-v1`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。
