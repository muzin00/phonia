# 最新SRC490モデルを固定した音素の組み合わせ選別

全36音素から一部を除き、本人照合のEERが改善するかを検証する。
`phoneme-corpus-expansion/src490-20261013-v1`の3音響seed、
Transformer seed 20261015・64k予算内の選定checkpointを固定する。
encoderもTransformerも再学習しない。既存コード・入力・モデルを変更しない。

## 事前に固定する比較

- 36音素すべてを選別対象にする。母音も個別に除外できるが、少なくとも1母音を残す。
- 従来の「入力には5母音必須」という採点条件と、登録話者全員に共通する音素maskを保持する。
  スコア計算に使用する音素だけを変えるので、候補間で本人・他人の試行と採点率は同じ。
- 各音素の登録・照合区間は固定し、短縮・再抽出・再配分をしない。
  選別で実使用音声量も減るため、固定総時間での比較ではない。
- 毎回Transformerのattentionとsoftmaxを再計算する。保存済み重みの再正規化ではない。
  maskに依存しない各tokenの入力projectionだけをcacheする。
  別経路の元モデルforwardと完全なスコア一致を実データで確認する。

## 話者を分けた選別

validationの各コーパス内で話者IDをSHA256順（seed 20261013）に並べ、
前半ceil(n/2)を探索、残りを確認にする。query・claimed speakerの両方が同じ側の照合だけを使う。
探索はJVS 8・CV 15・JTube 10話者、確認はJVS 7・CV 15・JTube 10話者。
登録区間と元の共通音素maskは、話者を分けても変えない。

通常文の3コーパス×3音響seedのEERを等重み平均し、全モデル共通の1集合を選ぶ。
まず全36音素から始め、1音素の全追加・全削除候補で最も改善するものへ移る。
改善が止まるか単独操作上限に達したら、除外済み音素と保持音素の全1対1入れ替えを試す。
入れ替え後は単独操作へ戻る。単独操作は最大8回、入れ替えは最大2回。
同値の移動はしない。複数の改善候補が同値なら音素inventoryのindex順で決める。
全候補スコアと不採用候補も保存する。これは全組み合わせの総当たりや大域最適性の証明ではない。

探索で選んだ1集合をhashとともに固定してから、確認側を一度だけ評価する。
確認の成績で集合を変えたり探索を追加したりしない。
この確認用話者は元のTransformer checkpoint選択には既に使われているため、
今回の音素探索からのみ分離した確認であり、モデル開発全体から独立したholdoutではない。

全validationを使ってコーパス・seed・集合ごとにFAR1%閾値を決め、全閾値を固定する。
その後、全36と選択1集合だけを同じtestで比較する。JVS別テキストは副結果。
確認で改善しなくても候補を差し替えず、その結果も報告する。

## 報告と限界

EER、FAR、未採点を含むFRR、採点率、seed別の対応差を保存する。
対応付き話者bootstrapは1,000回、query/claimed両軸を重み付けし、
同じコーパスでは全seedと両集合に同じ再標本化を使う。
区間は固定した集合・モデルに条件付いたもので、探索や学習の変動を含まない。
testは既観測で、新規話者・新規録音による一般化の確証ではない。
今回の結果で既存の標準モデル・照合設定を置き換えない。

## 実行

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-subset-selection/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-subset-selection
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-subset-selection

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-subset-selection/study.py execute
```

3 CPU worker・各1 threadで音響seedを分担する。中断後は同じコマンドで再開する。
固定ファイルが変わった場合は再開を拒否し、完了済み候補のスコアをhash確認して再利用する。
新しい出力先は[config.json](config.json)に事前固定する。

完了後の[結果](evaluation-results.md)と[全指標JSON](evaluation-results.json)に加え、
run内へ設計freeze、話者分割、全探索スコア、選択freeze、閾値、評価スコア、
対応付きbootstrap、元モデルとの一致・入力不変の監査を保存する。

完了後は `poc/phoneme-subset-selection/audit.py` で保存済み候補EER・探索手順・
全評価指標を独立に再集計する。

初回v1は140集合の探索・確認・test・bootstrap・レポート生成まで完了したが、
最後の進捗通知で引数名が重複する不具合があった。実行済みの旧コードと設定は
v1の`executed-source/`へ保存した。計算・選定規則を変えずに通知だけを修正し、
新規v2で全計算を再実行する。v2は結果の再現確認であり、独立した追加実験として数えない。
