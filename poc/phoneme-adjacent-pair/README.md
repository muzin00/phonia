# 隣接する子音→母音の結合

同じ発話内で実際に隣接するC/Vを、別々に特徴抽出する場合と連続区間として特徴抽出する場合で比較する。
元音声の利用範囲・総時間を揃え、元の境界をまたぐ自然な区間を使う。
今回は平均・標準偏差の特徴量を用いた最初の試作で、時間順序を保持するモデルは別の実験とする。

| 条件 | 特徴抽出 | encoderと集約 |
| --- | --- | --- |
| split | CとVそれぞれの64-bin log-mel平均・標準偏差 | 同じ共有MLPに個別入力し、L2正規化embeddingを等重み平均して再正規化 |
| joined | 元WAVのC開始からV終了までを一続きにした同じ特徴量 | 同じ構造のMLPへ入力してL2正規化 |

各条件を初期化から学習する。共有statistics MLPは65,920 parameters・128次元embedding。
計算上は両条件とも2 branchにし、joinedでは同じ特徴量を複製する。
branch間で同じdropout maskを使うため、joinedの2 branchは完全一致する。
初期値・乱数tensorのshape・学習token列を条件間で一致させる。音声を複製/連結する操作ではない。
差には区間単位での前処理、特徴抽出、embeddingの集約が含まれ、調音の遷移だけの因果効果とは解釈しない。

## 区間と対象ペア

SRC490と同じtrain 490ラベルの元データ、既存JVS/CV/JTubeのvalidation/test話者と元発話を使う。
選んだペアが存在する学習話者は489人。残るJTubeの1人には適格な対象区間がなく、両条件とも489クラスで学習する。
既存QCに合格した単音素区間から、アライメント番号が連続し境界が同じPCM frameのC→Vだけを採る。
休止、撥音N、促音clをC側に含めない。無声化母音の大文字は既存と同じ小文字ラベルへ対応させる。
各音素30ms以上・合計250ms以下とし、crop・padding・別発話の接ぎ合わせ・途中音素の飛び越しはしない。
両条件は厳密に同じPCM frameを使う。短い子音や長い組は対象外なので、全ての隣接音への結論にはならない。

ペア選択はEERを使わず、train/validationの件数だけで行う。

- 4つの学習コーパスそれぞれで、80%以上の話者に2区間以上ある。
- 3つのvalidationコーパスそれぞれで、全話者の登録元WAV集合に1区間以上ある。
- 各コーパスの通常文queryの5%以上に、元の5母音条件と当該ペアを満たす発話がある。
- 3コーパス中の最小query充足率が高い順に最大10種類。同率はラベル順。

119候補中15種類が条件を満たし、`d+e, k+a, n+a, n+i, n+o, r+u, sh+i, t+a, t+e, t+o`の10種類を固定した。
選択にtestの件数・音声・スコアは使わない。最適なペア集合の探索ではない。
学習用の実区間は288,266組。単音素の特徴量は元cacheと全件一致させ、結合特徴を元WAVから再抽出する。
特徴正規化は既存のtrain専用統計を両条件で共通に使う。

## 学習と評価

3 seeds × 2条件。各更新は5ペア種類group、各groupは10話者×2個の異なる実区間。
同じtokenを両条件へ同じ順序で与え、AAM＋0.5×同一ペア種類内SupConで学習する。
学習ラベルの人名やコーパスIDはモデル入力に含めない。
AdamW、LR 3e-4、warmup 1,000更新から1e-5へのcosine、weight decay 1e-4、clip 5。
1種類あたり6,000更新として計60,000更新、各ペア種類600,000回の提示。固定最終モデルを使い、validationでcheckpointを選ばない。
atomic checkpointへmodel・head・optimizer・乱数・履歴を一括保存し、同じscheduleで再開する。

登録は従来と同じ元WAV集合から、種類ごとに最大30組をhash順・元WAV分散で選ぶ。
queryも同じ元発話を使い、同じペア種類のembedding平均と登録平均のcosineを計算する。
queryに存在し全登録話者に揃う種類のcosineを等重み平均する。claimによって利用種類を変えない。
元の5母音query条件も保持し、利用可能なペアが0なら未採点として全入力FRRに残す。
両条件の登録区間、照合区間、本人/他人試行、採点率は同一。

全6モデルとコーパス/seed/条件ごとのvalidation FAR1%閾値を固定してから、test区間抽出と推論を行う。
通常文EER・固定閾値FAR/FRR・採点率・seed別差、JVS別テキストとペア別EERを保存する。
対応付きquery/claimed話者bootstrapは1,000回。同一コーパスではseed・条件間に同じ再標本化を使う。

これはペアのみを使う比較で、以前の全36音素Transformerとは入力・統合・学習・採点対象が異なる。
以前のEER値との直接比較や標準モデルの置換は行わない。
全testは既観測。CIは固定モデル・固定集合に条件付きで、学習/選択の変動を含まない。
ペア別EERは対象発話が異なるため、その順位を音素の優劣と扱わない。
人手の境界品質率や時間変化の有効性を検証した実験ではない。

## 実行

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-adjacent-pair/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-adjacent-pair
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-adjacent-pair

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-adjacent-pair/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-adjacent-pair/study.py status
```

既存のローカル音声・アライメント・特徴量が必要。3 CPU worker・各1 thread。
設定は[config.json](config.json)。同じexecuteで中断から再開し、完了済みならhashだけを確認する。
実験中に固定ファイルが変われば停止する。
区間の同一性、原音声からの特徴再計算、分離特徴全件、学習schedule・初期値・更新数、保存スコアのEER/FAR/FRRを監査する。

[結果](evaluation-results.md)・[全指標と監査](evaluation-results.json)。

初回v1は490人全員に対象ペアがあるという実装上の前提を検査して、model初期化・学習更新・test推論の前に停止した。
旧コードと設定はv1の`executed-source/`に保持し、有効489ラベルを明記したv2で実行する。
ペア選択・区間・特徴量・予算は変えず、再抽出した特徴量がv1と完全一致することを確認する。
