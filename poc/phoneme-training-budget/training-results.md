# 60,000回の学習量比較：学習・validation

同じ60,000回用schedule内の途中経過。1 seed・母音単一区間validation EER。

| コーパス | update | validation EER | 学習率 |
|---|---:|---:|---:|
| Common Voice | 30,000 | 9.175% | 0.00015886 |
| Common Voice | 45,000 | 9.228% | 0.00005384 |
| Common Voice | 60,000 | 9.136% | 0.00001000 |
| SRC4VC | 30,000 | 9.080% | 0.00015886 |
| SRC4VC | 45,000 | 9.026% | 0.00005384 |
| SRC4VC | 60,000 | 9.077% | 0.00001000 |

[学習曲線](learning-curves.svg)・[発話単位評価](evaluation-results.html)。途中の最良値で採用時点を変えない。
