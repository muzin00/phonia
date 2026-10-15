# EER改善を報酬にした学習データ選択

「この音素を含むデータを学習するとEERが改善／悪化した」を繰り返し、
次に学習するデータの分布を変える実験。既存の推論重み付けRLとは別の実装。
固定した学習済み音響encoderから得た特徴で、融合Transformerを未学習状態から学習する。
音響encoderそのものの初期学習をRLに置き換える実験ではない。

## 比較

| 条件 | 次の100更新で使うデータ |
| --- | --- |
| uniform | 候補から均等抽選 |
| random | 一様に選んだ行動で特定音素を含むペアを多めに抽選 |
| rl | EER改善の報酬で学習した方策が行動を選ぶ |

3音響seed × 3条件。各条件で同じ未学習Transformer、AdamW、BCE+uniform KLを使用し、
8,000更新 × 32発話（本人・他人の64照合）を継続する。
学習率3e-4、weight decay 0.01、KL係数0.001、勾配clip 5。dropoutは有効。
初期化seedと方策seedは各1種類。既存64,000更新モデルとの優位性比較ではない。

元の保存済み学習scheduleの先頭32,000行を4行ずつまとめ、1更新につき128発話の候補を作る。
元の他人claimを保持し、コーパス別の件数を候補の先頭32件と一致させる。
候補位置を重複なく32個選ぶ。元scheduleに同じ発話が複数あれば位置が異なる候補として扱う。
本人／他人のペアをセットで選ぶため、ラベル比率は全条件で一致する。

行動は均等抽選、または一つの音素を重視する抽選。
query・本人enrollment・他人enrollmentに共通する対象音素があれば抽選重みを4倍にする。
指数乱数キー `-log(U)/weight` の小さい順に各コーパスの定員まで選択する。
候補すべてで有無が同じ音素は行動から除外する。元の採点条件で必須の5母音はすべてに存在する。
**対象音素以外も入力に残す。推論では全条件が利用可能な全音素を使う。**
この実験が調べるのは音素を含む発話ペアの抽選効果であり、音素単体の普遍的な良し悪しではない。

## Teacherの強化学習

方策の入力は `[1, 学習進捗, 現在のreward macro EER]`。
行動数×3の線形パラメータをゼロで初期化し、softmax確率に20%の一様探索を混ぜる。
学習率0.03のAdam、uniform KL係数0.001、勾配clip 5を使用する。

1. 現在の方策から2行動を独立に抽選する。
2. 現在のstudentと**AdamWの全状態**を、均等抽選と抽選された各行動のために複製する。
3. 同じ候補、同じ抽選乱数、同じ更新別dropout seedで各分岐を100更新する。
4. 固定したreward話者に対する全音素推論の実測EERを求める。
5. `reward = EER(均等抽選の分岐) − EER(行動の分岐)` を計算する。
6. `mean(-reward.detach() * log π(action)) + λ KL(π || uniform)` でteacherを更新する。
7. **最初に抽選された分岐を必ず継続する。報酬が負でも継続する。**

同じ行動の分岐は完全に決定的なため結果を再利用する。
行動と独立した均等抽選の分岐がbaselineとなり、単なる学習進行との差を抑える。
最良の分岐を採用する貪欲探索ではない。1ブロック先の改善を学ぶcontextual banditであり、
長期の遅延報酬を最適化する多段RLや、大域最適を保証する方法ではない。
studentの継続更新数は一致するが、RLだけに追加の分岐計算が必要。実更新数も公開する。

## 分離と固定

元validation話者を、seed・コーパス・話者IDのSHA-256順で前半reward、後半selectionに分ける。
queryとclaimの**両方**がその集合に属する照合だけを使う。train/reward/selection/testは話者重複なし。
rewardはteacherの学習用であり、独立した検証集合として扱わない。
通常文のコーパス別EERを等重みで平均し、rewardとcheckpoint選定にそれぞれ使用する。

selectionで0更新から1,000更新おきに評価し、最小macro EERの最初のcheckpointを採用する。
最終8,000更新も別に報告。全モデルとselection FAR1%閾値を固定してからtest推論を行う。
JVSの別文条件は副次評価。EERは採点可能分、FAR/FRRは採点不能を拒否として含む。

既存データと旧実験は変更しない。全testは以前の実験で観測済み。
音響encoderの旧選定に元validationが使われた可能性があり、完全な未観測ベンチマークではない。
少数のreward話者への過学習や、teacherの学習変動には注意が必要。
95%区間は固定モデルを条件とする対応付き話者bootstrap（1,000回）で、
学習・checkpoint選定の変動は含まない。標準モデルは自動更新しない。

## 再現と検証

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-training-data-rl/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-training-data-rl
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-training-data-rl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-training-data-rl/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-training-data-rl/study.py status
```

初回にコード・設定・入力・乱数・初期モデル・話者分割をハッシュ固定。
各音響seedのworkerを並行実行し、100更新ごとに原子的checkpointを保存する。
再実行で完了済み条件を検証してスキップし、途中条件は最後のブロックから再開する。
再開後のdropoutと抽選は保存済み乱数／更新番号から決まり、通常実行と完全一致する。
凍結後に設定や実装を変える場合は、新しいrunディレクトリで別実験にする。

テストはコーパス件数・本人他人比率・全音素入力の保持、抽選効果、期待報酬の厳密勾配、
分岐のoptimizer分離、dropoutの一致、負報酬でも第1分岐を継続すること、
3条件の完全再開、同値時の最初のcheckpoint選定、話者分割を確認する。
実行後監査では保存された抽選位置と音素出現数、teacher更新を全ブロック再計算し、
推論の直接forwardとの一致、独立したEER実装と閾値固定FAR/FRRを照合する。

背景は [Learning to Teach (ICLR 2018)](https://www.microsoft.com/en-us/research/publication/learning-to-teach/)
のteacher/studentによるデータ選択。今回のEER差分報酬と対照分岐はこの実験固有の設計であり、論文の忠実な再現ではない。

結果は実行後に `evaluation-results.md` / `evaluation-results.json` に出力する。
