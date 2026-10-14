# EER近似を加えたTransformer追加学習

SRC490・36音素、固定encoderの3音響seed平均。initialは追加学習前、BCEは対照、soft-EERはBCE＋KL＋EER近似。
両学習条件とも同じ出発モデル、同じ128発話/256照合batch、同じ2,000更新。selectedはvalidation選定、finalは固定終点。

## EER

| 集合 | コーパス | 追加前 | BCE選定 | soft-EER選定 | BCE最終 | soft-EER最終 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | jvs | 0.411% | 0.411% | 0.411% | 0.547% | 0.544% |
| validation | common_voice | 3.199% | 3.199% | 3.199% | 3.731% | 3.878% |
| validation | jtube | 0.835% | 0.835% | 0.835% | 0.987% | 0.987% |
| validation | macro | 1.482% | 1.482% | 1.482% | 1.755% | 1.803% |
| test | jvs | 1.001% | 1.001% | 1.001% | 1.189% | 1.195% |
| test | common_voice | 2.556% | 2.556% | 2.556% | 2.222% | 2.232% |
| test | jtube | 0.929% | 0.929% | 0.929% | 1.183% | 1.041% |
| test | macro | 1.495% | 1.495% | 1.495% | 1.531% | 1.489% |

## soft-EER − BCEのtest EER差

| コーパス | checkpoint | 差 | 改善 / 同値 / 悪化seed | 対応付き95%区間 |
| --- | --- | ---: | --- | --- |
| jvs | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| jvs | final | +0.006 pp | 1 / 0 / 2 | [-0.071, +0.111] pp |
| common_voice | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| common_voice | final | +0.010 pp | 1 / 1 / 1 | [-0.230, +0.154] pp |
| jtube | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| jtube | final | -0.142 pp | 2 / 1 / 0 | [-0.399, +0.083] pp |
| macro | selected | +0.000 pp | 0 / 3 / 0 | [+0.000, +0.000] pp |
| macro | final | -0.042 pp | 3 / 0 / 0 | [-0.157, +0.055] pp |

## validation FAR1%閾値を固定したtest性能

| コーパス | 条件 | FAR | 全入力FRR | 採点率 |
| --- | --- | ---: | ---: | ---: |
| jvs | initial | 0.724% | 3.333% | 98.000% |
| jvs | bce-selected | 0.724% | 3.333% | 98.000% |
| jvs | soft_eer-selected | 0.724% | 3.333% | 98.000% |
| jvs | bce-final | 0.673% | 3.911% | 98.000% |
| jvs | soft_eer-final | 0.670% | 3.911% | 98.000% |
| common_voice | initial | 0.852% | 41.370% | 64.556% |
| common_voice | bce-selected | 0.852% | 41.370% | 64.556% |
| common_voice | soft_eer-selected | 0.852% | 41.370% | 64.556% |
| common_voice | bce-final | 0.963% | 38.037% | 64.556% |
| common_voice | soft_eer-final | 0.937% | 38.481% | 64.556% |
| jtube | initial | 1.035% | 40.667% | 59.833% |
| jtube | bce-selected | 1.035% | 40.667% | 59.833% |
| jtube | soft_eer-selected | 1.035% | 40.667% | 59.833% |
| jtube | bce-final | 1.140% | 40.667% | 59.833% |
| jtube | soft_eer-final | 1.114% | 40.611% | 59.833% |
| macro | initial | 0.870% | 28.457% | 74.130% |
| macro | bce-selected | 0.870% | 28.457% | 74.130% |
| macro | soft_eer-selected | 0.870% | 28.457% | 74.130% |
| macro | bce-final | 0.925% | 27.538% | 74.130% |
| macro | soft_eer-final | 0.907% | 27.668% | 74.130% |

## 選択された追加更新数

| 音響seed | 出発点の旧更新 | BCE | soft-EER |
| --- | ---: | ---: | ---: |
| 20260926 | 58000 | 0 | 0 |
| 20260927 | 22000 | 0 | 0 |
| 20260928 | 49000 | 0 | 0 |

EERは採点可能分、FAR/FRRは未採点を拒否として含む。testのFARを同じ値に揃えた比較ではない。全条件で採点対象を固定。

近似損失の温度0.03・係数1・追加2,000更新の単一条件。論文EERLossの再現ではない。3つは音響seedで、追加学習seedは1つ。全testは既観測。CIは固定モデルを条件とした話者bootstrapで、学習や選択の変動を含まない。今回の結果だけで標準モデルを置き換えない。

成果物: `artifacts/phoneme-eer-finetuning/src490-softeer-20261015-v1`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。
