# 新規Common Voice話者での母音不足対応

判定: **adoption_deferred**。事前固定した研究用基準の結果は {'far_upper95_at_most_1pct': False, 'far_increase_upper95_at_most_0_1pp': False, 'frr_improvement_upper95_below_zero': True}。

学習に使っていない60 client_idを校正30・test30に固定。各話者20発話で登録し、30発話を照合。全36音素encoder・等重み・登録上限30区間を固定。testを見て閾値や条件を選び直していない。

## test全入力の性能

| 閾値 | 条件 | 採点可能 | EER | FAR | FRR | 他人誤受入 |
| --- | --- | --- | --- | --- | --- | --- |
| calibrated | vowels5 | 581/900 | 3.959% | 0.441% | 44.222% | 115 |
| calibrated | vowels4 | 819/900 | 4.396% | 0.640% | 23.444% | 167 |
| calibrated | vowels3_phones5 | 889/900 | 4.724% | 0.628% | 18.444% | 164 |
| jvs_transfer | vowels5 | 581/900 | 3.959% | 5.092% | 36.889% | 1329 |
| jvs_transfer | vowels4 | 819/900 | 4.396% | 7.165% | 11.444% | 1870 |
| jvs_transfer | vowels3_phones5 | 889/900 | 4.724% | 7.713% | 4.111% | 2013 |

## 主比較の信頼区間

| 指標 | 差または値 | 95%区間 |
| --- | --- | --- |
| 4母音−5母音 far_1pct/far | +0.199 pp | [+0.071, +0.402] pp |
| 4母音−5母音 far_1pct/frr | -20.778 pp | [-24.889, -17.000] pp |
| 4母音−5母音 eer | +0.437 pp | [-0.300, +1.102] pp |
| 4母音FAR | 0.640% | [0.160, 1.524]% |

## 欠損別の内訳（新校正閾値・目標FAR 1%）

| 条件 | 欠損母音 | 話者数 | 本人入力 | 採点可能 | 本人拒否 | 他人誤受入/試行 |
| --- | --- | --- | --- | --- | --- | --- |
| vowels5 | a | 6 | 6 | 0 | 6 | 0/174 |
| vowels5 | ae | 2 | 2 | 0 | 2 | 0/58 |
| vowels5 | aeo | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | ai | 2 | 2 | 0 | 2 | 0/58 |
| vowels5 | aie | 2 | 2 | 0 | 2 | 0/58 |
| vowels5 | au | 4 | 5 | 0 | 5 | 0/145 |
| vowels5 | aue | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | auo | 2 | 2 | 0 | 2 | 0/58 |
| vowels5 | e | 29 | 83 | 0 | 83 | 0/2407 |
| vowels5 | eo | 11 | 12 | 0 | 12 | 0/348 |
| vowels5 | i | 11 | 13 | 0 | 13 | 0/377 |
| vowels5 | ie | 2 | 3 | 0 | 3 | 0/87 |
| vowels5 | ieo | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | io | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | iu | 4 | 4 | 0 | 4 | 0/116 |
| vowels5 | iue | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | iueo | 2 | 2 | 0 | 2 | 0/58 |
| vowels5 | none | 30 | 581 | 581 | 79 | 115/16849 |
| vowels5 | o | 20 | 40 | 0 | 40 | 0/1160 |
| vowels5 | u | 28 | 96 | 0 | 96 | 0/2784 |
| vowels5 | ue | 7 | 11 | 0 | 11 | 0/319 |
| vowels5 | ueo | 1 | 1 | 0 | 1 | 0/29 |
| vowels5 | uo | 14 | 30 | 0 | 30 | 0/870 |
| vowels4 | a | 6 | 6 | 6 | 3 | 1/174 |
| vowels4 | ae | 2 | 2 | 0 | 2 | 0/58 |
| vowels4 | aeo | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | ai | 2 | 2 | 0 | 2 | 0/58 |
| vowels4 | aie | 2 | 2 | 0 | 2 | 0/58 |
| vowels4 | au | 4 | 5 | 0 | 5 | 0/145 |
| vowels4 | aue | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | auo | 2 | 2 | 0 | 2 | 0/58 |
| vowels4 | e | 29 | 83 | 83 | 15 | 22/2407 |
| vowels4 | eo | 11 | 12 | 0 | 12 | 0/348 |
| vowels4 | i | 11 | 13 | 13 | 4 | 5/377 |
| vowels4 | ie | 2 | 3 | 0 | 3 | 0/87 |
| vowels4 | ieo | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | io | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | iu | 4 | 4 | 0 | 4 | 0/116 |
| vowels4 | iue | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | iueo | 2 | 2 | 0 | 2 | 0/58 |
| vowels4 | none | 30 | 581 | 581 | 78 | 119/16849 |
| vowels4 | o | 20 | 40 | 40 | 8 | 8/1160 |
| vowels4 | u | 28 | 96 | 96 | 22 | 12/2784 |
| vowels4 | ue | 7 | 11 | 0 | 11 | 0/319 |
| vowels4 | ueo | 1 | 1 | 0 | 1 | 0/29 |
| vowels4 | uo | 14 | 30 | 0 | 30 | 0/870 |
| vowels3_phones5 | a | 6 | 6 | 6 | 3 | 1/174 |
| vowels3_phones5 | ae | 2 | 2 | 2 | 0 | 0/58 |
| vowels3_phones5 | aeo | 1 | 1 | 0 | 1 | 0/29 |
| vowels3_phones5 | ai | 2 | 2 | 2 | 0 | 0/58 |
| vowels3_phones5 | aie | 2 | 2 | 0 | 2 | 0/58 |
| vowels3_phones5 | au | 4 | 5 | 5 | 3 | 0/145 |
| vowels3_phones5 | aue | 1 | 1 | 0 | 1 | 0/29 |
| vowels3_phones5 | auo | 2 | 2 | 0 | 2 | 0/58 |
| vowels3_phones5 | e | 29 | 83 | 83 | 15 | 19/2407 |
| vowels3_phones5 | eo | 11 | 12 | 12 | 4 | 4/348 |
| vowels3_phones5 | i | 11 | 13 | 13 | 4 | 5/377 |
| vowels3_phones5 | ie | 2 | 3 | 3 | 2 | 0/87 |
| vowels3_phones5 | ieo | 1 | 1 | 0 | 1 | 0/29 |
| vowels3_phones5 | io | 1 | 1 | 1 | 1 | 0/29 |
| vowels3_phones5 | iu | 4 | 4 | 4 | 0 | 0/116 |
| vowels3_phones5 | iue | 1 | 1 | 0 | 1 | 0/29 |
| vowels3_phones5 | iueo | 2 | 2 | 0 | 2 | 0/58 |
| vowels3_phones5 | none | 30 | 581 | 581 | 80 | 112/16849 |
| vowels3_phones5 | o | 20 | 40 | 40 | 8 | 7/1160 |
| vowels3_phones5 | u | 28 | 96 | 96 | 24 | 11/2784 |
| vowels3_phones5 | ue | 7 | 11 | 11 | 4 | 0/319 |
| vowels3_phones5 | ueo | 1 | 1 | 0 | 1 | 0/29 |
| vowels3_phones5 | uo | 14 | 30 | 30 | 7 | 5/870 |

## 限界と監査

30 test話者を共有する2,000回の対応付きbootstrap。固定モデル・閾値に条件付いた区間で、校正の不確実性や多重比較補正を含まない。元コーパスは学習と同じCommon Voiceだがclient_idは分離。実在人物の重複と録音sessionは不明で、別日・別端末の保証ではない。

監査: {'score_checks': 54000, 'rate_checks': 144, 'bootstrap_checks': 960, 'float64_max_error': 1.4912690660118244e-07, 'status': 'passed'}

[固定設計と再実行](README.md)・[全指標JSON](evaluation-results.json)・[結論](interpretation.md)
