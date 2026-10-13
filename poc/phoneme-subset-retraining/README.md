# 34音素に合わせたTransformerの再学習

前回の[音素選別](../phoneme-subset-selection/interpretation.md)では、固定モデルの照合時に
`k`・`ch`を除外するとtest平均EERが1.495→1.421%へ下がったが、確認側の平均改善は残らなかった。
今回は、この34音素に合わせて統合層を初期化から学習すると改善が安定するかを調べる。
新しい音素探索はせず、候補は前回の固定集合1つだけにする。

| 条件 | Transformerの学習 | 照合 |
| --- | --- | --- |
| train36_score36 | 既存36音素 | 36音素 |
| train36_score34 | 同じ既存36音素モデル | k/ch除外 |
| train34_score34 | 初期化からk/ch除外で新規学習 | k/ch除外 |

## 揃える条件

- 既存SRC490の3音響seed、固定encoderと登録/queryのembedding、学習データ。
  encoderは490ラベルで学習済み、統合層は従来どおり入力条件を満たす485ラベル。
- Transformer seed 20261015。同じ初期値から学習し、以前の学習済みTransformerから継続しない。
  元の36 token位置と36ラベルembeddingを保持し、初期化とdropout乱数のshapeを変えない。
- 保存済み64k更新分のquery・負例ペア列をそのまま使う。先頭4kはランダム負例、以後は既存のCV限定難例。
  34音素用に負例を再選定しない。本人/他人ペアで同じ音素maskを使う。
- 1更新32 query・本人/他人64照合、AdamW、BCE＋一様重みへのKL、gradient clip。
- 最初4kのcosine LR（3e-4から約3e-5）、4k以後は3e-5固定、合計64k。
- 5母音必須の採点条件、共通登録音素mask、入力音声・登録上限・採点対象は不変。

新規学習は3本。変更するのは学習/validation時にもk/chを除外する点。
除外音素はattentionのkey maskと最終softmaxの両方から外し、損失にも寄与させない。

## 選定と評価

従来と同じく、更新0と500更新ごとの3コーパス全validation通常文EERの等重み平均を使い、
64k以内の最小・最初の同値を選ぶ。主結果はこの選定モデル、副結果は固定最終64kモデル。
旧36音素モデルは既存の選定/最終状態を使い、新規学習と同じ予算で比較する。

36→34の対照は、36音素validationで選ばれた既存checkpointを保持する。
旧全checkpointを34音素向けに選び直す実験ではない。
したがって主結果には学習とcheckpoint選択の違いが含まれる。固定最終64kで更新数の違いを取り除いた比較も示す。
前回の探索/確認用validationも今回はcheckpoint選択に使うため、独立holdoutとは呼ばない。

全モデルと、各条件・コーパスの全validation FAR1%閾値を固定してからtestを採点する。
EER、FAR/FRR、採点率、seed別対応差、JVS別テキストの副結果を保存する。
旧条件のスコアを独立再推論し、元の実験・前回の音素選別結果との完全一致を確認する。

EER差にはquery/claimed両軸の対応付き話者bootstrap 1,000回を付ける。
同じコーパスでは全seed・条件に共通の再標本化を使う。
区間は固定モデルと固定音素集合を条件とし、学習・選択の変動を含まない。
全testは既観測であり、新規母集団への汎化確認ではない。標準設定の置換は行わない。

## 検証と再開

本学習前に、最初の音響seedの旧36音素条件を4kまで再現し、model・optimizer・乱数・
全validation履歴・選定モデルが既存成果物と完全一致することを要求する。
モデル、optimizer、乱数、履歴、bestモデルを1つのatomic checkpointに保存する。
中断時に異なる保存時点のbestを混ぜず、同じ学習ペアとdropout乱数で再開する。
単体テストで中断/再開と連続学習の完全一致、学習ペアと特徴の不変、除外音素の勾配0を検査する。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-subset-retraining/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-subset-retraining
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-subset-retraining

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-subset-retraining/study.py execute
# 同じコマンドで中断後も再開できる。完了時はhash確認だけを行う。
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-subset-retraining/study.py status
```

3 CPU worker・各1 thread。設定は[config.json](config.json)。
完了後の[結果](evaluation-results.md)・[全指標と監査](evaluation-results.json)を参照する。
