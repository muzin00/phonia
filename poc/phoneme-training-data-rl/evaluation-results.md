# EER改善を報酬にした学習データ選択

学習済みの音響encoderを固定し、Transformerを同じ未学習状態から学習。3音響seed平均。
uniformは均等抽選、randomはランダムな音素重視、rlは報酬に基づく音素重視。各条件8,000更新、32発話/64照合。
100更新ごとに2行動を抽選し、同じモデル・AdamW状態から分岐した均等抽選との差で報酬を計算。最初の行動を必ず継続。
selectedは独立したselection話者で選定、finalは8,000更新の終点。推論時は全条件で利用可能な全音素を使う。

## EER

| 集合 | コーパス | initial | uniform-selected | random-selected | rl-selected | uniform-final | random-final | rl-final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | jvs | 0.607% | 0.255% | 0.340% | 0.158% | 0.049% | 0.255% | 0.267% |
| selection | common_voice | 3.741% | 2.128% | 2.229% | 2.229% | 2.128% | 2.026% | 2.229% |
| selection | jtube | 4.286% | 1.693% | 1.587% | 1.799% | 2.152% | 2.222% | 2.240% |
| selection | macro | 2.878% | 1.359% | 1.385% | 1.395% | 1.443% | 1.501% | 1.579% |
| test | jvs | 0.862% | 1.053% | 0.739% | 0.904% | 1.046% | 0.998% | 1.497% |
| test | common_voice | 3.672% | 2.760% | 2.944% | 3.436% | 2.960% | 2.914% | 2.861% |
| test | jtube | 1.955% | 1.212% | 1.319% | 0.929% | 1.603% | 1.280% | 1.578% |
| test | macro | 2.163% | 1.675% | 1.667% | 1.756% | 1.870% | 1.731% | 1.979% |

## Test EERの対応差

| コーパス | 比較 | 差 | 改善/同値/悪化seed | 対応付き95%区間 |
| --- | --- | --- | --- | --- |
| jvs | rl-selected-versus-uniform-selected | -0.149 pp | 2 / 0 / 1 | [-0.673, +0.035] pp |
| jvs | rl-selected-versus-random-selected | +0.165 pp | 1 / 0 / 2 | [-0.189, +0.324] pp |
| jvs | random-selected-versus-uniform-selected | -0.314 pp | 2 / 0 / 1 | [-0.894, +0.070] pp |
| jvs | rl-final-versus-uniform-final | +0.450 pp | 1 / 0 / 2 | [-0.133, +0.927] pp |
| jvs | rl-final-versus-random-final | +0.499 pp | 0 / 0 / 3 | [-0.034, +1.006] pp |
| jvs | random-final-versus-uniform-final | -0.049 pp | 2 / 0 / 1 | [-0.304, +0.147] pp |
| common_voice | rl-selected-versus-uniform-selected | +0.677 pp | 0 / 0 / 3 | [+0.223, +1.200] pp |
| common_voice | rl-selected-versus-random-selected | +0.493 pp | 1 / 0 / 2 | [-0.034, +1.040] pp |
| common_voice | random-selected-versus-uniform-selected | +0.184 pp | 1 / 1 / 1 | [-0.168, +0.625] pp |
| common_voice | rl-final-versus-uniform-final | -0.099 pp | 1 / 1 / 1 | [-0.733, +0.402] pp |
| common_voice | rl-final-versus-random-final | -0.053 pp | 1 / 0 / 2 | [-0.505, +0.342] pp |
| common_voice | random-final-versus-uniform-final | -0.046 pp | 2 / 0 / 1 | [-0.420, +0.427] pp |
| jtube | rl-selected-versus-uniform-selected | -0.283 pp | 2 / 1 / 0 | [-1.160, +0.242] pp |
| jtube | rl-selected-versus-random-selected | -0.391 pp | 2 / 0 / 1 | [-0.838, +0.042] pp |
| jtube | random-selected-versus-uniform-selected | +0.108 pp | 1 / 0 / 2 | [-0.534, +0.437] pp |
| jtube | rl-final-versus-uniform-final | -0.024 pp | 1 / 1 / 1 | [-0.316, +0.170] pp |
| jtube | rl-final-versus-random-final | +0.298 pp | 0 / 0 / 3 | [-0.075, +0.641] pp |
| jtube | random-final-versus-uniform-final | -0.323 pp | 3 / 0 / 0 | [-0.735, +0.046] pp |
| macro | rl-selected-versus-uniform-selected | +0.081 pp | 1 / 0 / 2 | [-0.306, +0.299] pp |
| macro | rl-selected-versus-random-selected | +0.089 pp | 1 / 0 / 2 | [-0.176, +0.330] pp |
| macro | random-selected-versus-uniform-selected | -0.008 pp | 1 / 0 / 2 | [-0.317, +0.208] pp |
| macro | rl-final-versus-uniform-final | +0.109 pp | 2 / 0 / 1 | [-0.204, +0.350] pp |
| macro | rl-final-versus-random-final | +0.248 pp | 0 / 0 / 3 | [-0.056, +0.453] pp |
| macro | random-final-versus-uniform-final | -0.139 pp | 2 / 0 / 1 | [-0.326, +0.085] pp |

## Selection FAR1%閾値を固定したtest性能

| コーパス | 条件 | FAR | 全入力FRR | 採点率 |
| --- | --- | --- | --- | --- |
| jvs | initial | 0.698% | 2.889% | 98.000% |
| jvs | uniform-selected | 1.032% | 3.022% | 98.000% |
| jvs | random-selected | 1.213% | 2.400% | 98.000% |
| jvs | rl-selected | 0.978% | 2.711% | 98.000% |
| jvs | uniform-final | 1.175% | 3.022% | 98.000% |
| jvs | random-final | 1.194% | 2.933% | 98.000% |
| jvs | rl-final | 1.165% | 3.689% | 98.000% |
| common_voice | initial | 0.286% | 46.444% | 64.556% |
| common_voice | uniform-selected | 1.662% | 37.704% | 64.556% |
| common_voice | random-selected | 1.642% | 38.296% | 64.556% |
| common_voice | rl-selected | 1.799% | 38.519% | 64.556% |
| common_voice | uniform-final | 1.451% | 38.852% | 64.556% |
| common_voice | random-final | 1.738% | 37.704% | 64.556% |
| common_voice | rl-final | 1.782% | 37.296% | 64.556% |
| jtube | initial | 1.409% | 41.222% | 59.833% |
| jtube | uniform-selected | 1.921% | 40.278% | 59.833% |
| jtube | random-selected | 2.047% | 40.389% | 59.833% |
| jtube | rl-selected | 1.830% | 40.444% | 59.833% |
| jtube | uniform-final | 2.284% | 40.611% | 59.833% |
| jtube | random-final | 2.254% | 40.556% | 59.833% |
| jtube | rl-final | 2.240% | 40.667% | 59.833% |
| macro | initial | 0.798% | 30.185% | 74.130% |
| macro | uniform-selected | 1.538% | 27.001% | 74.130% |
| macro | random-selected | 1.634% | 27.028% | 74.130% |
| macro | rl-selected | 1.536% | 27.225% | 74.130% |
| macro | uniform-final | 1.636% | 27.495% | 74.130% |
| macro | random-final | 1.729% | 27.064% | 74.130% |
| macro | rl-final | 1.729% | 27.217% | 74.130% |

## 学習・探索

| 音響seed | 条件 | 選定更新 | 実際のstudent更新 | 正/負のrollout報酬数 | 負報酬でも継続した回数 |
| --- | --- | --- | --- | --- | --- |
| 20260926 | uniform | 5000 | 8000 | 0 / 0 | 0 |
| 20260926 | random | 4000 | 8000 | 0 / 0 | 0 |
| 20260926 | rl | 3000 | 23200 | 86 / 69 | 33 |
| 20260927 | uniform | 8000 | 8000 | 0 / 0 | 0 |
| 20260927 | random | 5000 | 8000 | 0 / 0 | 0 |
| 20260927 | rl | 4000 | 23000 | 75 / 78 | 38 |
| 20260928 | uniform | 5000 | 8000 | 0 / 0 | 0 |
| 20260928 | random | 5000 | 8000 | 0 / 0 | 0 |
| 20260928 | rl | 5000 | 23100 | 88 / 66 | 31 |

## RLの学習終了時の行動確率

| seed | 上位5行動（進捗1・最終reward EERで算出） |
| --- | --- |
| 20260926 | m: 10.03%, gy: 5.65%, d: 4.83%, r: 4.80%, cl: 4.69% |
| 20260927 | py: 6.01%, n: 4.95%, ny: 4.51%, ch: 4.47%, z: 4.45% |
| 20260928 | gy: 6.71%, v: 6.31%, m: 5.28%, N: 5.07%, h: 4.97% |

報酬用話者はteacherの学習データ。selection/testとは話者を分離し、queryとclaimの両側で制限した。
音素は学習データの抽選にのみ利用し、入力音素を削除しない。5母音は候補すべてに存在するため抽選行動から除外。
EERは採点可能分、FAR/FRRは未採点を拒否として含む。全条件で採点対象は固定。test FARは同値に揃えていない。
RLは単一ブロックの改善を学ぶcontextual banditで、長期最適化や大域最適を保証しない。対照より追加の分岐計算を使う。
3音響seed・1初期化/方策seed・単一設定。既存64kモデルへの優位性を調べる実験ではない。全testは既観測。
CIは固定モデルを条件とした話者bootstrapで学習・選択の変動を含まない。既存encoderの選定に旧validationが使われた可能性も残る。標準モデルは置き換えない。

成果物: `artifacts/phoneme-training-data-rl/src490-data-rl-20261015-v1`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。
