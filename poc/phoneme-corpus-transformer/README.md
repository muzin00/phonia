# 3コーパスencoderとTransformerの統合

JVS・Common Voice・JTubeSpeech-ASVで学習した280ラベルencoderを固定し、音素ごとの照合情報をTransformerで統合するとEERが改善するかを検証する。音声区間のencoderをTransformerに置き換える実験ではなく、既存の128次元embeddingと音素別類似度から照合スコアを作る統合層の比較である。

先行する[Transformer統合](../phoneme-transformer-fusion/README.md)のネットワーク実装を再利用する。先行実験や元のコーパス拡張実験のコード・入力・モデルは変更しない。利用者の希望により、未採点原因の追加調査よりこの検証を優先する。新規録音は不要。

## 結果（2026-10-12）

6モデルすべて4,000更新まで学習し、validationで固定したcheckpointのtest評価を完了した。**TransformerはJTubeのEERを約36%下げたが、Common Voiceでは改善せず、一律の置換モデルとはしない。**

| 評価コーパス | 等重み EER | Transformer EER（3seed平均±SD） | 平均の差 | 改善seed数 |
| --- | ---: | ---: | ---: | ---: |
| JVS | 1.497% | 1.348±0.138% | −0.149ポイント | 2/3（残りは同率） |
| Common Voice | 4.303% | 4.362±0.041% | ＋0.059ポイント | 0/3 |
| JTube | 1.965% | 1.256±0.376% | −0.709ポイント | 3/3 |

音響encoderを再学習せず、照合情報の統合方法を変えるだけでJTubeの改善が3seedとも得られた。JVSの改善は小さく、Common Voiceは全seedでわずかに悪化した。SDは統合層seed間のばらつきで、話者母集団の信頼区間ではない。

MLPは3seedともvalidationで更新0（等重み）が選ばれたため、採用checkpointのtest指標は対照と同じだった。学習自体は全4,000更新を実行したが、この設計では学習済みMLPを選ぶ根拠が得られなかった。Transformerの選択更新はseed順に3,500・3,000・3,500。

| 評価コーパス | 等重み FAR→Transformer平均 | 等重み FRR→Transformer平均 | 採点率（共通） |
| --- | ---: | ---: | ---: |
| JVS | 0.714%→0.806% | 4.133%→4.133% | 98.00% |
| Common Voice | 0.195%→0.497% | 51.667%→55.704% | 64.56% |
| JTube | 0.640%→0.868% | 42.333%→40.889% | 59.83% |

FAR/FRRはそれぞれのモデルのvalidation FAR1%で決めた閾値をtestに適用した値で、test FARを同値に揃えた比較ではない。CVは本人拒否が平均約4.04ポイント増加しており、EERの小幅悪化だけで問題を過小評価しない。採点率は完全に同じなので、未採点減少による改善ではない。

JVSの文章を変えた追加評価ではEERが1.403%から1.160%（平均）へ下がったが、1seedは1.531%へ悪化した。通常評価を主結果とし、この追加評価を選択には使っていない。

次は、既存の別seedの280ラベルencoderでも同じ比較を行い、JTube改善とCV悪化が音響encoderの初期化に依存するかを確認するのが妥当。今回のtest結果を見て選択規則や閾値を後から変えず、次の条件は別runとして先に固定する。

## 比較条件

- encoderは初回コーパス拡張実験の`expanded280`、音響学習seed 20260926。ファイルSHA256は`023499dea430342f9071b799c7607f78867680448d9d2c0b171378137fb72ddc`。学習時も評価時も更新しない。
- 対照は従来の音素別スコアの等重み平均。MLPとTransformerをそれぞれ3seed（20261015・20261016・20261017）、各4,000更新で学習する。各seedで同じ照合ペアを使う。
- 1更新は32 queryと各queryに対する本人・同コーパス他人の64照合。本人・他人で共通の音素maskを使い、欠落パターンだけで正解を判別できないようにする。
- 入力は登録embedding、query embedding、差の絶対値、積、双方の区間数・平均区間長・embeddingの散らばり・RMS。話者ID・コーパスID・正解ラベルはネットワーク入力に含めない。
- Transformerは隠れ次元64、2層、4head。音素IDを使い、時間位置は与えない。MLPは音素を独立に処理する対照。いずれも音素ごとの重みをsoftmaxで求め、元の類似度の加重平均を最終スコアとする。
- 3コーパスのvalidation EERを等重みで平均し、最小のcheckpointを選ぶ。同率は先の更新を選び、等重み状態の更新0も候補に含める。testを見て学習条件・checkpoint・閾値を変更しない。
- 全6モデルと各コーパスのvalidation FAR1%閾値を固定してからtestを採点する。

## 入力の分離

元のtrain manifest内で、ラベルごとにWAVのSHA256を固定seedで並べ、先頭`min(20, 異なるWAV数の半分切り捨て)`を登録用、残りを照合用とする。登録は音素ごとに最大30区間、異なるWAVを順番に使う。5母音がそろう登録とqueryがあるラベルだけを統合層の学習に使う。

統合層の学習対象はJVS70・CV70・JTube139の279ラベル、15,123 query。`jtube_bej`は利用可能な異なるWAVが1本だけのため除外した。元のencoderは280ラベル学習のままである。別音声による補充はしていない。

validation/testは元の拡張280評価の登録・照合インデックスをそのまま使う。学習との話者ラベル・WAV hashの重複を検査する。5母音必須条件、共通登録音素、未採点の扱いを維持し、全6評価splitで等重みの再計算スコアと保存済みスコアの差が1e-12未満であることを学習前に確認した。

EERは採点可能分、FAR/FRRは未採点を拒否として含む全入力の値。既存の5音響seed平均との直接比較はせず、同一の固定encoderの等重み対照と比較する。今回の3seedは統合層の初期化と照合ペア順の違いであり、音響encoderの再学習ではない。全testは既観測なので、新規話者集団での確証や運用精度の証明とは扱わない。

## 実行

設定は[config.json](config.json)。成果物は`artifacts/phoneme-corpus-transformer/frozen280-20261012-v1`へ保存する。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-transformer/experiment.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-transformer/experiment.py status
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-transformer/audit.py
```

`execute`はコード・入力hashの固定、登録/queryの準備、学習、選択の固定、評価まで実行する。500更新ごとに進捗と再開用checkpointを保存する。同時実行はロックで防ぐ。中断後も同じコマンドで再開できる。

監査は学習登録/queryのWAV分離、保存した学習順の再生成、checkpoint選択、元の対照結果との一致、全照合の再推論、採点率・閾値・指標の再計算を行う。入力変換の3テストと既存ネットワークの3テストを通過してから学習を開始した。

完了後の監査も通過した。validation/test合計798,000照合レコードを照合し、学習済み統合モデルの再推論スコアの最大差は0。固定encoder、学習順、選択checkpoint、採点可否、閾値と集計値の整合性を確認した。

完了時の`results.json`と`results.md`に等重み・MLP・Transformerの結果を保存し、`complete-audit.json`に監査結果を保存する。
