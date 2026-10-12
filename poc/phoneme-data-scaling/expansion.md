# 確認後の全量処理と学習比較

2026-10-12、利用者から「OK 確認した進めて」と指示されたため、今回のレビュー対象に対応する固定11,200発話の処理と学習比較を開始した。個別回答や品質率を自動生成せず、全件回答・保存は求めない。

## 全量処理

対象は[選定した4集合](README.md)。原音声は11,200件（Common Voice 7,000、JTube 4,200）で、追加話者や発話の差し替えは行わない。レビューで取得済みの160件とJTube書き起こしを照合して再利用する。新規JTubeの取得は4接続まで、ASRは既存のローカルWhisperモデル、抽出は3プロセスで実行する。

処理条件はレビューと同じ。Common Voiceの配布文、JTubeの自動生成テキストを使い、24kHz mono PCM16、Julius 10秒上限、発話30秒上限、30〜250ms、RMS −50dBFS以上、既存36音素を適用する。空の音素列・時間上限・位置合わせ不成立は使用不可として記録し、代替発話を選ばない。未知の実行失敗はパイプラインを停止させ、品質上の失敗へ黙って読み替えない。

全入力のreceipt、ASRとの対応、全特徴量行と区間manifestの一致を監査する。既存学習・評価の区間で参照される元波形と、使用可能区間を持つ追加発話について、WAVヘッダーによらないPCM bytesのSHA-256を比較する。既存波形または別の追加発話と一致する追加発話は全条件から除外する。実在人物のコーパス間同一性は保証しない。

## 学習と評価の固定方針

- 比較条件は現状280、発話追加280、話者追加420候補、両方追加420候補。後二者の実際の学習可能話者数はQC・重複除外後に記録する。
- 原280条件のmanifestと特徴量を全条件の先頭に保持する。対照条件を再構成した時に元ファイルとのhash一致も確認する。
- 音響encoderは既存と同じstatistics MLP、seed `20260926`、216,000更新、同じ既存train由来の特徴量統計。追加3条件を同一の初期encoderから学習し直し、現状条件は凍結済みモデルを再利用する。
- Transformerは既存と同じ2層・hidden 64・4 heads、3 seeds `20261015/16/17`。最初の4,000更新のcosine学習率、以降3e-5固定、計16,000更新。既存16k比較と同じschedule生成規則を使う。
- validationの3コーパスEER平均が最小のcheckpointを主結果とし、同値なら早い更新を選ぶ。最終16k checkpointも副結果として残す。
- すべてのモデルを固定後、コーパス別のvalidation FAR1%閾値を固定し、既存と全く同じtest入力を評価する。新しい条件のtest scoreはこの順序まで計算しない。
- EERに加えてFAR・FRR・coverageを保持する。音響encoder 1 seed、統合層3 seedsの探索比較であり、全testは既観測。純粋な話者数の因果効果や未知データへの頑健性の証明とは扱わない。

前処理後に4条件のmanifest、話者支持数、コード、学習設定を`design-freeze.json`へ固定してから学習する。全量処理前に書いた設定は学習方針であり、未完了データから学習を開始しない。学習環境でも各追加集合200区間の特徴量を再計算し、最大差2e-6以内を確認する。

## 実行と再開

```sh
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/acquire_expansion.py
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/expansion_pipeline.py
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/continue_expansion.py
```

`continue_expansion.py`は前処理の完了を待ち、全量監査→特徴量の互換性確認→3条件のencoder・Transformer学習→比較評価→結果監査を順に実行する。3条件は並列、各条件内のTransformer 3 seedsは順番に処理する。原280の16k結果は再現一致を検査する。処理の途中停止はreceiptやcheckpointから再開し、凍結済み入力を変更しない。

| 保存先 | 主な状態ファイル |
| --- | --- |
| `artifacts/phoneme-data-scaling/expanded-inputs-20261012-v1` | `download-progress.json`, `asr-progress.json`, `extraction-{0,1,2}-progress.json`, `pipeline-state.json` |
| `artifacts/phoneme-data-scaling/comparison420-20261012-v1` | `continuation-state.json`, 各条件の`state.json`, `results.json`, `results.md`, `complete-audit.json` |

完了後の監査では、共通の評価入力、学習内の登録・照合波形の分離、16kまでのschedule、checkpoint選定、閾値、保存された全評価スコアを再検算する。途中の学習lossや採点数を改善の結論として扱わない。

## 前処理・学習入力の確定結果

全11,200件のreceiptと特徴量行の対応を監査済み。10,633発話で使用可能区間が得られ、225発話は使用可能区間なし、342発話は30秒上限による除外だった。失敗を補充していない。

| 追加集合 | 固定発話 | 区間を得た発話 | 音素区間 | 区間を持つ話者ラベル |
| --- | ---: | ---: | ---: | ---: |
| 既存CVの発話追加 | 3,500 | 3,490 | 156,530 | 40 |
| 既存JTubeの発話追加 | 2,100 | 1,935 | 94,575 | 137 |
| 新規CV話者 | 3,500 | 3,500 | 138,151 | 70 |
| 新規JTube話者 | 2,100 | 1,708 | 101,506 | 70 |
| 合計 | 11,200 | 10,633 | 490,762 | — |

追加の使用可能発話10,633件について、既存学習・評価の参照元32,006波形との完全一致、および追加発話同士の完全一致は0件。WAVヘッダーを除いたPCMで比較した。近似コピーや実在人物の重複を排除したという意味ではない。

| 条件 | 学習話者ラベル | 音素区間 |
| --- | ---: | ---: |
| baseline280 | 280 | 1,269,961 |
| utterances280 | 280 | 1,521,066 |
| speakers420 | 420 | 1,509,618 |
| combined420 | 420 | 1,760,723 |

4条件で予定ラベルの欠落は0。元280条件のmanifest・特徴量の再構成は元ファイルとhash一致した。追加4集合から200区間ずつ、計800区間を学習環境で再計算した特徴量の最大差は全集合で0だった。レビューと全量ASRのモデルbytes・パッケージ・デコード設定の一致も確認済み。

Transformerの学習対象はbaseline280・utterances280が279ラベル、speakers420・combined420が415ラベル。後二者では登録・照合に異なる音声を持たない2ラベル、5母音を含む照合がない1ラベル、登録に母音が不足する2ラベルが対象外となった。encoderは予定どおり280/420ラベルを使用した。Transformerの照合例数は順に15,123・19,134・17,283・21,294件。

## 学習・評価結果

2026-10-12、追加3条件のencoder各216,000更新と、各条件3 seedsのTransformer各16,000更新を完了した。主結果は事前に決めたvalidation選定checkpoint。以下のEERは採点可能なtest照合に対する値で、Transformer欄は3 seedsの平均±標準偏差（パーセント表示）。標準偏差は音響encoderのseed差や未知話者集合の不確実性を含まない。

| 条件 | JVS EER | Common Voice EER | JTube EER |
| --- | ---: | ---: | ---: |
| 現状280 | 1.108 ± 0.221% | 3.905 ± 0.401% | 1.129 ± 0.280% |
| 発話追加280 | 1.678 ± 0.614% | 4.109 ± 0.277% | 1.300 ± 0.322% |
| 話者追加420 | 1.513 ± 0.326% | 4.077 ± 0.051% | 1.031 ± 0.119% |
| 両方追加420 | 1.150 ± 0.069% | 3.840 ± 0.264% | 0.801 ± 0.277% |

両方追加はJTubeで0.327ポイント（相対29.0%）改善した。対応する3 seeds中2 seedsで改善、1 seedで悪化しており、全seedで一貫する改善ではない。Common Voiceは0.065ポイントの小幅改善（改善1・同値1・悪化1 seed）、JVSは0.042ポイント悪化（改善1・悪化2 seeds）。発話追加だけでは3コーパスすべてで平均EERが悪化した。今回の範囲では、既存話者の発話追加より新しい話者も加える方向に改善の手掛かりがあるが、データ量に対する単調な改善は確認できなかった。

### Encoder単体との比較

| 条件 | JVS EER | Common Voice EER | JTube EER |
| --- | ---: | ---: | ---: |
| 現状280 | 1.497% | 4.303% | 1.965% |
| 発話追加280 | 1.497% | 4.647% | 2.228% |
| 話者追加420 | 1.497% | 3.733% | 1.407% |
| 両方追加420 | 1.224% | 3.614% | 1.730% |

両方追加のencoder単体は現状より3コーパスとも改善した。一方、そこにTransformerを加えた時の変化はJVS 1.224→1.150%、Common Voice 3.614→3.840%、JTube 1.730→0.801%。Transformerによる追加改善は一様ではなく、Common Voiceではencoder単体より悪化した。JVSが比較的良好な録音環境という利用者の観察と併せ、コーパス別に判断する。今回の比較だけで差の原因を録音品質と特定はできない。

### 実運用の閾値と最終checkpoint

コーパス・モデルごとにvalidationの採点可能分でFAR 1%を狙った閾値を固定した。次のtest FAR/FRRは未採点を拒否扱いにした全照合の分母、3 seeds平均。testで同じFARを保証する閾値ではない。

| コーパス | 現状 FAR → 両方追加 FAR | 現状 FRR → 両方追加 FRR | 採点率（全条件共通） |
| --- | ---: | ---: | ---: |
| JVS | 0.762 → 0.663% | 3.556 → 3.556% | 98.000% |
| Common Voice | 0.512 → 0.688% | 53.222 → 45.000% | 64.556% |
| JTube | 0.909 → 1.351% | 40.778 → 40.611% | 59.833% |

Common Voiceの本人拒否率は8.222ポイント低下したが、誤受入率も増えている。JTubeはEERが下がっても固定閾値での誤受入率が増え、1%を上回った。したがって、そのまま認証用モデルを置き換える結論にはしない。評価入力と採点条件が固定のため未採点率は変わらず、これを解消した実験でもない。

副結果として最終16k checkpointのEER平均も保存した。

| 条件 | JVS EER | Common Voice EER | JTube EER |
| --- | ---: | ---: | ---: |
| 現状280 | 1.195% | 4.060% | 1.222% |
| 発話追加280 | 1.442% | 4.131% | 1.486% |
| 話者追加420 | 1.425% | 3.808% | 1.578% |
| 両方追加420 | 1.224% | 3.828% | 1.090% |

両方追加のJTube改善は最終checkpointでも見られるが、話者追加だけのJTubeは最終checkpointでは悪化した。都合のよいcheckpointへ主結果を変更せず、事前指定のvalidation選定を維持する。

各条件の音響encoderは1 seed、Transformerは3 seedsで、test集合はすべて既観測。今回の結果は探索的な改善傾向であり、スケーリング則や統計的有意性の確認ではない。次に確かめるなら、現状と両方追加のencoderを別seedでも学習し、今回の改善が音響encoderの初期値に依存しないかを見る。学習量との相互作用を調べる場合は、その後に拡張データで更新数を増やす比較を別実験として固定する。

数値の全件表は`artifacts/phoneme-data-scaling/comparison420-20261012-v1/results.md`、seed別値と閾値は同じディレクトリの`results.json`へ保存した。

## 完了監査と成果物

2026-10-12 13:42 JSTに独立監査まで完了した。3,192,000件の評価レコードを検証し、Transformerスコアの独立再計算との最大差は0。閾値・EER/FAR/FRR、評価入力の全条件一致、学習内の登録・照合波形分離、16kまでのschedule、checkpoint選定、凍結hashを確認した。現状280のselected/final結果は前回の16k実験と完全一致した。監査結果は`complete-audit.json`、自動継続処理の完了は`continuation-state.json`に保存済み。

主結果の選定更新数は以下のとおり。すべての条件で16,000更新まで学習したうえでvalidationから選んだ値であり、testによる選び直しではない。

| 条件 | seed 20261015 | seed 20261016 | seed 20261017 |
| --- | ---: | ---: | ---: |
| 現状280 | 12,500 | 6,000 | 13,000 |
| 発話追加280 | 12,000 | 7,500 | 11,000 |
| 話者追加420 | 9,500 | 15,500 | 13,000 |
| 両方追加420 | 7,500 | 16,000 | 12,000 |

[比較グラフ](../../artifacts/phoneme-data-scaling/comparison420-20261012-v1/data-scaling-eer.png)はencoder単体とTransformerの主結果を表示する。再生成は次のコマンドで行う。監査未完了のrunからは生成しない。

```sh
MPLCONFIGDIR=/tmp/phonia-expansion-mpl \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-data-scaling/report_expansion.py
```

同時に`comparison-summary.json`へ平均・標準偏差・最終checkpoint平均・対応seedの改善数を保存する。
