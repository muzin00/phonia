# データ量と学習更新数の交互作用・encoder seedの再現性

前回の[話者・発話拡張](../phoneme-data-scaling/expansion.md)では、両方追加420のencoder単体は3コーパスで改善したが、Transformer追加後の改善はJTubeに偏った。利用者の「次はその観点で進めて」を受け、追加データ取得を行わず、同じ確認済みデータで更新数と音響学習seedの影響を比較する。

## 固定する比較

| 要因 | 条件 |
| --- | --- |
| 学習データ | baseline280 / combined420（元manifest・特徴量・統計をそのまま利用） |
| encoder seed | 20260926 / 20260927 / 20260928 |
| encoder更新数 | 216,000 / 432,000 |
| Transformer seed | 20261015に固定 |
| Transformer更新数 | 16,000 / 64,000 |

encoderの3 seedsは初期値と学習サンプリングの両方が変わる。各seedをデータ2条件に対応させ、同じ初期encoderで比較する。Transformer seedは全条件で固定し、今回の平均・標準偏差は音響学習seed間で計算する。前回の「固定encoder・Transformer 3 seeds平均」と今回の平均を同じ統計量として直接比較しない。

音響encoderは元の216,000更新までのcosine学習を保持し、その最終モデル・分類head・AdamW・乱数状態から432,000更新まで継続する。以降の学習率は終端1e-5に固定する。サンプリングは追加216,000更新を別block（元seed+1,000,003）で生成し、既存の学習順を変えない。長い予算に合わせたcosineの引き伸ばしや学習率の再上昇は行わない。encoderは両予算の最終状態を評価する。

各encoderに対するTransformerは元の4,000更新のcosine、その後3e-5固定で64,000更新まで学習し、16,000更新時点も保存する。更新0を含む各予算内の等コーパスvalidation EER平均最良checkpointを主結果、最終checkpointを副結果とする。既存seedの216k encoderと既存280の64k Transformerは監査済み成果物を再利用する。他のTransformerは同じ4kの学習経路から継続し、既存420の16k結果との一致を監査する。

全12 encoder条件とTransformerの全checkpointを固定してから校正し、全validation閾値固定後に新しいtestスコアを計算する。評価用話者、登録・照合音声、採点規則は前回と同一。FAR1%閾値はコーパス・モデルごとにvalidationの採点可能分で決め、test FAR/FRRは未採点を拒否に含めて報告する。既観測testによる探索比較であり、新しい分布への汎化保証にはしない。

## 判断する点

- 同じ更新数で280→420に増やした差が、3つの音響学習seedで再現するか。
- データごとの216k→432k encoder更新の効果、およびデータ追加の効果が更新数で変わるか。
- 同じencoderに対するTransformer 16k→64kの効果。encoder単体の改善を保持できるか。
- 主結果と最終checkpointの差、EERと固定閾値FAR/FRRのトレードオフ。

1つの終盤学習率での追加更新を検証するため、改善がなくてもモデル容量や別の学習率方針を含む「データスケーリング全般の限界」とは結論しない。今回の3 seedsでは統計的有意性も断定しない。

## 実行

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-data-budget/data_budget.py execute
```

3 workerで6つのデータ×音響seedの学習を進め、その後に全条件の校正・test評価・独立監査・集計を実行する。同じコマンドでcheckpointから再開可能。監査はencoderのスコア再計算、Transformerの独立再推論、学習順・optimizer更新数・checkpoint選定・閾値・既存結果の再現を含む。

成果物は`artifacts/phoneme-data-budget/replicated280420-20261012-v1`。`state.json`と条件ごとの`state.json`、学習履歴に進捗を保存する。完了後に`results.md`・`comparison-summary.json`・`data-budget-eer.png`・`complete-audit.json`を保存する。

2026-10-12 14:00 JSTに実行を開始。新規2テスト（encoderの中断再開後のモデル・head・optimizer完全一致、継続サンプリングの再現性と異なる正例ペア）と、既存Transformer継続の3テストが成功した。最初のsandbox実行は既存checkpointのMPS乱数状態を復元できず停止したため、14:02 JSTに以前と同じネイティブ環境で再開した。CPUで学習し、凍結済みコード・学習条件・入力は変更していない。途中まで進んだ新seedも保存済みcheckpointから再開した。

## 学習・評価の集計

2026-10-12 16:25 JSTに全学習、16:32 JSTに全条件の校正・test集計を完了した。今回の主結果は**encoderの3 seeds平均・Transformer固定1 seed**である。前回の固定encoder・Transformer 3 seeds平均とは分母が異なるため、今回の280条件を対照にする。以下はvalidation選定モデルの採点可能分EER。全体の認証失敗率ではない。

### データ追加の効果

まず元と同じencoder 216k・Transformer 16k予算を比較した。

| 評価 | 280 EER（平均±SD） | 420 EER（平均±SD） | 相対低下 | 対応seedで改善 |
| --- | ---: | ---: | ---: | ---: |
| JVS | 1.616 ± 0.546% | 1.331 ± 0.182% | 17.6% | 2/3 |
| Common Voice | 4.408 ± 0.842% | 3.152 ± 0.598% | 28.5% | 2/3 |
| JTube | 1.231 ± 0.103% | 0.914 ± 0.416% | 25.8% | 2/3 |

データ追加で3コーパスとも平均EERが改善した。いずれも全seedで改善する結果ではないが、encoder単体のCommon Voice・JTubeでは216k/432kの両予算で3/3 seedsが改善した。216k encoder単体はCV 4.386→3.474%、JTube 1.955→1.769%。今回の280→420の範囲では、データ拡張が有効という傾向が前回より明確になった。

### 420条件の追加更新

| encoder更新 | Transformer予算 | JVS EER | Common Voice EER | JTube EER |
| ---: | ---: | ---: | ---: | ---: |
| 216,000 | 16,000 | 1.331% | 3.152% | 0.914% |
| 432,000 | 16,000 | 1.367% | 3.337% | 0.782% |
| 216,000 | 64,000 | 1.199% | 3.298% | 1.021% |
| 432,000 | 64,000 | 1.237% | 3.567% | 0.929% |

encoderを2倍学習するとJTubeは0.914→0.782%へ低下したが、JVS・CVは上昇した。Transformerを16k→64kへ延長すると、420では両encoder予算ともJVSが改善し、CV・JTubeの平均EERは悪化した。JTubeのTransformer延長は216k encoderで改善0・同値1・悪化2 seeds、432k encoderで改善0・同値2・悪化1 seedだった。

追加学習でデータ拡張の効果が一律に強くなる結果でもなかった。データ効果を「420 EER − 280 EER」と定義し、216k encoder/16k Transformerからの変化を示す。負はデータ追加の有利さが広がったことを表す。単位はパーセントポイント、対応する音響seed間の差を平均した値。

| 更新数の変更 | JVS | Common Voice | JTube |
| --- | ---: | ---: | ---: |
| encoderのみ432kへ | +0.159 | +0.295 | −0.137 |
| Transformerのみ64kへ | +0.049 | +0.524 | −0.039 |
| 両方を432k/64kへ | +0.126 | +0.696 | +0.088 |

したがって「前回の拡張条件が単に学習不足だった」という説明だけでは結果全体を説明できない。データ拡張には効果があるが、同じ終盤学習率で更新数を増やせば全体が伸びるという結果ではない。

### Checkpoint選定と運用指標

420条件の最終checkpoint（事前指定の副結果）も確認した。

| encoder更新 | Transformer最終更新 | JVS EER | Common Voice EER | JTube EER |
| ---: | ---: | ---: | ---: | ---: |
| 216,000 | 16,000 | 1.513% | 3.427% | 1.021% |
| 432,000 | 16,000 | 1.477% | 3.438% | 1.119% |
| 216,000 | 64,000 | 1.406% | 3.310% | 1.114% |
| 432,000 | 64,000 | 1.286% | 3.098% | 1.207% |

CVは最終checkpointではTransformer追加更新で改善する一方、主結果のvalidation選定モデルでは改善していない。特に432k/64kは主結果3.567%、最終状態3.098%。この違いは学習更新数だけでなく、等コーパスvalidation EERによるcheckpoint選定とtest結果の関係を調べる余地を示す。testを見て主結果を最終checkpointへ切り替えたり、コーパス別の最良値を単一モデルの性能としてまとめたりしない。これだけでTransformerの容量不足・過学習・録音品質を原因と特定もしない。

運用指標は同じ216k/16k予算で280→420に変えると、CV FRRが52.852→44.481%、FARが0.525→0.609%。JTube FRRが40.778→40.611%、FARが0.936→1.184%。420の432k/16kではJTube EER 0.782%でもFAR 1.213%、FRR 40.500%だった。validationで決めた閾値がtestでFAR 1%を保証する結果ではない。採点率は全条件共通でJVS 98.000%、CV 64.556%、JTube 59.833%。

## 結論と次の検証

データを増やす方針を断念する結果ではない。今回の範囲では新しい話者と発話の追加が有効であり、CV・JTubeのencoder単体では別seedでも改善が揃った。一方、更新数を一律に増やすことを次の標準設定にはしない。

次に検証するなら、既存成果物でvalidationとtestの傾向、checkpoint選定の相性を確認する。CVの選定モデルと最終モデルの差を調べることで、追加データの効果を統合層・選定方法で失っていないかを絞る。新たな学習データや本人の録音は、この確認には必要ない。

3つの音響seedと1つの固定Transformer seed、既観測testでの探索結果であり、統計的有意性や大規模なスケーリング則の証明とはしない。別の学習率スケジュールやモデル容量についても未検証である。

## 完了監査

2026-10-12 16:44 JST、学習・評価・独立監査・集計をすべて完了した。6,840,000件の評価レコードを検証し、Transformerの独立再推論スコアの最大差は0。encoderのvalidation/testスコアも全条件で再計算し、完全一致した。

- 3 seedsそれぞれで280/420の初期encoder状態が一致し、seed間では異なることを確認。
- 元216kの学習順hashと追加216kの学習順を再生成。432k時点のoptimizer更新数・終端学習率・exportとのモデル一致を検証。
- Transformerの64kまでの学習順、予算内checkpoint選定、validation閾値、test指標を再検算。
- 全条件の評価入力は同一。元seedの216k encoder/16k Transformerの結果は前回と完全一致。
- 新規2テストはsandbox・ネイティブ両環境で成功。既存Transformer継続3テスト、Ruff check/format checkも成功。

数値の全表は[results.md](../../artifacts/phoneme-data-budget/replicated280420-20261012-v1/results.md)、seed別EERと対応する差は[comparison-summary.json](../../artifacts/phoneme-data-budget/replicated280420-20261012-v1/comparison-summary.json)、図は[比較グラフ](../../artifacts/phoneme-data-budget/replicated280420-20261012-v1/data-budget-eer.png)、監査は[complete-audit.json](../../artifacts/phoneme-data-budget/replicated280420-20261012-v1/complete-audit.json)。音声・モデル・スコア・図はGit管理外。
