# 全36音素モデルでの母音不足対応

全36音素encoder・登録上限30区間/音素と、既存のclean学習Transformer 3 seedを固定し、照合入力の受理条件だけを変更する。新規学習・音声加工・音素境界変更は行わない。

## 事前固定する条件

| 条件 | 必要な母音種類数 | 母音・子音を合わせた必要種類数 |
| --- | ---: | ---: |
| 5母音必須（既存対照） | 5 | 5 |
| 4母音以上（主比較） | 4 | 4 |
| 3母音以上＋合計5種類以上（補助比較） | 3 | 5 |

合計5種類は、母音を減らした場合にも最低5種類の比較材料を要求する探索条件であり、5母音と同じ情報量や安全性を意味しない。利用音素は従来どおり、各splitの全登録話者に存在する学習済み音素と照合入力との共通部分。存在する音素はすべて統合する。話者候補ごとに受理条件を変えない。

主比較は等重み・4母音以上と5母音必須の全入力FAR/FRR差。Transformer 3 seedと混合音素条件は補助比較とする。Transformerは学習時に5母音が揃う入力だけを使用しており、母音不足への適用は未知の入力条件に対する診断となる。testによるseed・条件の選び直しは行わない。

各モデル・条件のvalidation通常文でFAR 1%・FAR 0.1%・EER動作点を校正し、全閾値を固定してからtestを計算する。元の5母音必須の閾値を据え置いた結果も併記し、既存入力の判定変化と新規に採点可能になった入力を分ける。無採点は全入力FAR/FRRで拒否として数える。EERは採点可能入力だけの診断値で、条件間で母集団が変わる。

欠損母音パターン別の入力数・話者数・誤受入件数・本人拒否件数を出す。15 test話者の対応付き2,000回bootstrapでFAR/FRR/EER/coverage差を測る。固定モデル・固定閾値に条件付いた区間で、学習・閾値校正・発話内容の再抽出や多重比較の不確かさは含めない。FAR差の区間が0を含んでも同等性の証明とは扱わない。

元の5母音必須のスコア・指標・閾値との一致、追加スコアのembeddingからの独立再構成、全指標の再計算、モデル・入力hashの不変を確認する。既存の観測済みJVS testを使う探索で、新しい独立holdoutや別日・別端末の評価ではない。

## 実行

リポジトリrootから実行する。既存artifactが必要。`prepare`は未使用の出力先だけを許可する。再実行時は全コマンドに同じ新規`--run`を指定する。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-missing-vowel-evaluation/experiment.py prepare
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-missing-vowel-evaluation/experiment.py calibrate
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-missing-vowel-evaluation/experiment.py evaluate
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-missing-vowel-evaluation/auditing.py
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-all-missing-vowel-evaluation/report.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-all-missing-vowel-evaluation/tests -v
```

[測定表](evaluation-results.md)・[HTML](evaluation-results.html)・[JSON](evaluation-results.json)・[解釈](interpretation.md)
