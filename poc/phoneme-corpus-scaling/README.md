# 第三コーパスと学習話者数の拡張

利用者は、本人識別の可能性を確認できた段階で新規録音を終了し、JVS・Common Voiceに偏らない学習データ拡張による改善傾向の確認を希望した。現行all36はJVS 70話者＋Common Voice 70 clientラベルの140ラベル。**2026-10-11に固定済み9,000発話の準備と3条件の追加学習・評価が完了した。JTubeの未学習話者では改善傾向が得られたが、Common Voiceは悪化しており、全体改善とは判定しない。**

次のモデル比較として、利用者の希望する[280ラベルencoderとTransformerの統合](../phoneme-corpus-transformer/README.md)を実施した。初回の280ラベルencoderを固定した3統合seedで、TransformerのJTube EERは1.965%→平均1.256%へ改善した。一方、CVは4.303%→4.362%、本人拒否は51.667%→55.704%へ悪化し、全体改善とは判定しない。以下の5音響seed平均とは条件が異なる。

## 初回比較の結果

全条件は同じ初期encoder・216,000更新・最終checkpointを使用した。対照は既存モデルを再利用。EERは採点可能な照合での誤り率で、低いほどよい。

| 学習条件 | JVS EER | Common Voice EER | JTube EER |
| --- | ---: | ---: | ---: |
| 対照140（JVS70・CV70） | 1.769% | 3.959% | 8.782% |
| 混合140（JVS70・CV35・JTube35） | 1.448% | 5.543% | 4.662% |
| 拡張210（JVS70・CV70・JTube70） | 1.370% | 4.457% | 3.856% |
| 拡張280（JVS70・CV70・JTube140） | 1.497% | 4.303% | 1.965% |

JTubeのEERは280条件で対照より約78%低下した。未採点を含めた本人拒否率（FRR）は62.67%→42.33%、差は−20.33ポイント（話者単位paired bootstrap 1,000回の95%区間：−28.17〜−12.66ポイント）。他人受入率（FAR）は0.439%→0.640%で、差の95%区間は0を含む。校正群のFAR1%で定めた閾値をtestに適用している。

Common Voiceは280条件でFRRが44.22%→51.67%、差は＋7.44ポイント（95%区間：＋4.67〜＋10.22ポイント）となり、FAR低下と本人拒否増加が同時に起きている。JVSのFRR差は−0.27ポイントで95%区間は0を含む。したがって追加コーパスへの適応を示す有力な初回結果だが、既存コーパスを含めた一律の改善ではない。

採点率は全モデル共通でJVS98.00%、CV64.56%、JTube59.83%。未採点は拒否に含めるため、JTubeには約40%のFRR下限が残る。モデルの分離性能と、必要な音素が揃って照合できる割合を分けて扱う。1 seed・JTube test20ラベルの探索結果であり、複数seedの再現性や実録音の運用性能は未確認。録音環境・ASR誤り・話者数・音声量の効果も分離できていない。

この初回結果を受けて、以下の複数seed検証と学習投入比率の介入を実施した。テスト結果を見て今回の閾値やcheckpointを変更しない。

### 5seedによる再現検証

初回の20260926に固定seed 20261012・20261013・20261014・20261015を加えた検証を[独立した実行コード](../phoneme-corpus-seed-replication/replicate.py)で実施し、2026-10-11に完了した。元のコード・入力・結果は変更していない。追加seedごとに基準140も再学習し、各seed内の同じ初期encoderから混合140・拡張210・拡張280と比較した。音声の再抽出はせず、4条件の固定済みキャッシュをhash確認して再利用した。

最初に初回基準モデルを再学習し、最終encoderと学習投入順のhashが元の結果と完全一致した。その後、新規16モデルすべてを216,000更新まで学習してから評価を開始した。閾値は各モデル・各コーパスの校正群だけで再計算し、最終checkpoint・5母音条件・未採点を拒否とする規則を維持した。全seedを採点し、最良seedの選択はしていない。

結果は`artifacts/phoneme-corpus-seed-replication/five-seeds-20261011-v1`へ保存した。平均・標本SD・範囲、各seedの基準との差と改善方向の一致数を報告する。今回の全testは既観測であり、複数seedは学習初期化・投入順に対する感度検証である。独立の話者・録音を増やした検証や母集団の信頼区間とは扱わない。

EERの5seed平均±標本SDは次のとおり。SDはseed間のばらつきで、信頼区間ではない。

| 条件 | JVS EER | Common Voice EER | JTube EER |
| --- | ---: | ---: | ---: |
| 基準140 | 1.328±0.278% | 4.042±0.143% | 7.404±1.027% |
| 混合140 | 1.326±0.170% | 5.171±0.309% | 4.260±0.283% |
| 拡張210 | 1.203±0.240% | 4.196±0.256% | 3.862±0.400% |
| 拡張280 | 1.384±0.265% | 4.147±0.242% | 1.900±0.076% |

**JTubeの改善は5/5seedで再現した。** 拡張280のEERは基準より平均5.504ポイント低く、平均値の相対低下は約74%。各seedのEERは1.803〜1.965%と安定している。本人拒否率は平均60.300%→42.067%（−18.233ポイント）で、これも5/5seedで低下した。一方、他人受入率は0.381%→0.716%へ5/5seedで増加したため、全指標が同時に改善したとは言わない。

**Common Voiceは「EERが常に悪化」ではなく「本人拒否増加が安定して再現」と判断を更新する。** 拡張280のEER差は平均＋0.106ポイントだが、改善2seed・悪化3seedで方向が揃わない。本人拒否率は44.089%→51.378%（＋7.289ポイント）、各seedの差も＋5.778〜＋8.222ポイントで5/5seed悪化した。他人受入率は0.394%→0.198%へ低下しており、校正後の拒否側への偏りが課題として残る。拡張210も本人拒否が平均＋3.422ポイント、5/5seed悪化した。CV学習ラベルを半減させる混合140はEER・本人拒否とも5/5seed悪化した。

JVSは拡張280のEER差が平均＋0.056ポイント、改善2・同値1・悪化2seedで一貫した改善を示さない。拡張210は4/5seedでEER低下したが、平均差は−0.124ポイントにとどまる。採点率はどのseed・条件でもJVS98.00%、CV64.56%、JTube59.83%で変わらない。追加学習だけでは音素不足などの未採点を解決できていない。

新規テスト3件、Ruff検査を通過した。[完了監査](../phoneme-corpus-seed-replication/audit.py)では20モデル、60校正閾値、testの1,140,000照合レコードを確認し、全入力を含むFAR/FRR・採点率と平均/SD・seed内の差を照合した。元の凍結コードと入力hash、評価入力の全seed一致も確認済み。

全指標と各seedの値は[集計JSON](../../artifacts/phoneme-corpus-seed-replication/five-seeds-20261011-v1/results.json)、[結果表](../../artifacts/phoneme-corpus-seed-replication/five-seeds-20261011-v1/results.md)、[監査結果](../../artifacts/phoneme-corpus-seed-replication/five-seeds-20261011-v1/completion-audit.json)を参照。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-corpus-seed-replication/tests
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-seed-replication/replicate.py freeze
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-seed-replication/replicate.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-seed-replication/replicate.py status
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-seed-replication/audit.py
```

### 学習投入比率の介入

5seed検証をPR #74としてマージ後、利用者の継続指示により[投入比率の検証](../phoneme-corpus-balance/run.py)を実施し、2026-10-11に学習・評価・監査まで完了した。280ラベル・音声・特徴量・5seed・初期encoder・損失・216,000更新・音素ごとの投入枠を固定し、話者の抽出方法だけを変更した。元の学習コードを編集せず、今回の実行中だけ固定した[sampler](../phoneme-corpus-balance/sampler.py)を注入した。元の280条件は各seedの完了済みモデルを再利用し、140・210条件も参考として併記する。

比較条件はJVS:CV:JTubeを1:1:1とする`balanced280`、1:2:1とする`cvweighted280`の2種類。音素ごとに最大10話者×2区間を取り、同一グループ内の話者重複・同一区間の正例ペアを作らない。利用可能な話者の少ないコーパスの割当は他へ回し、全体の有効区間数と各音素の有効区間数が元の280条件と一致することを検査する。比率は希少音素では達成できないため、元の方式も含めたコーパス別・音素別の実投入数を保存する。

2条件×5seedの全10モデルを学習した後、校正群のFAR1%から全閾値を固定してtestを採点した。主比較は同seedの元の280条件に対するCV本人拒否率の低下と、JTube EERの変化。全条件・全seed・FARも報告し、最良seedやtestを見た閾値を選んでいない。今回も全testは既観測であり、独立データへの汎化や録音環境だけの因果効果は示さない。

5seedの平均値は次のとおり。本人拒否率には未採点を含む。

| 評価・指標 | 元の280 | 均等280 | CV重点280 |
| --- | ---: | ---: | ---: |
| CV EER | 4.147% | 4.348% | 3.955% |
| CV 本人拒否率 | 51.378% | 47.267% | 43.267% |
| CV 他人受入率 | 0.198% | 0.350% | 0.614% |
| JTube EER | 1.900% | 1.759% | 1.956% |
| JTube 本人拒否率 | 42.067% | 41.433% | 41.833% |
| JTube 他人受入率 | 0.716% | 0.923% | 0.989% |
| JVS EER | 1.384% | 1.876% | 2.091% |
| JVS 本人拒否率 | 4.133% | 5.227% | 4.667% |
| JVS 他人受入率 | 0.691% | 0.920% | 1.735% |

**CVへの投入を増やすと本人拒否は減ったが、全体改善には至っていない。** 均等条件のCV本人拒否は5/5seedで低下（平均−4.111ポイント）した一方、CV EERは4/5seedで悪化した。CV重点条件は本人拒否が5/5seedで低下（平均−8.111ポイント、各seed −10.222〜−6.333ポイント）、EERも4/5seedで低下した。ただしCVの他人受入は両条件とも5/5seedで増加している。

均等条件のJTube EERは4/5seedで低下し、CV重点条件は平均＋0.056ポイントと元の280に近い。一方、CV重点条件のJVS EERと他人受入率は5/5seedで悪化した。JVS testの他人受入率は平均1.735%となり、校正群で狙った1%を超えている。追加コーパスで得た分離性能を保てる候補ではあるが、全コーパス・他人受入まで含めた置換モデルとして採用する根拠はない。

最初のseedの実投入割合はJVS/CV/JTubeの順に、元の方式34.25/23.69/42.06%、均等条件34.85/32.86/32.29%、CV重点26.78/49.00/24.22%。希少音素の制約により目標比率と差がある。全seed・全条件の有効投入は21,060,000区間で、音素別の総投入も一致した。採点率は変化せず、音素不足などによる未採点の課題も残る。

この結果を受けて、以下の保存済みスコアによる校正移行・同FAR上限の診断を実施した。分離性能の改善と判定が通りやすくなった影響を調べ、testから運用閾値は選ばない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-corpus-balance/tests
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-balance/run.py freeze
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-balance/run.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-balance/run.py status
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-balance/audit.py
```

出力先は`artifacts/phoneme-corpus-balance/five-seeds-20261011-v1`。サンプラーの新規テスト3件とRuff検査を通過した。[完了監査](../phoneme-corpus-balance/audit.py)は比較対象を含む25モデル、75校正閾値、test 1,425,000照合を検証し、2条件の学習投入順も全216,000更新分を再構成してhashの一致を確認した。全指標・SD・seed内差は[集計JSON](../../artifacts/phoneme-corpus-balance/five-seeds-20261011-v1/results.json)、[結果表](../../artifacts/phoneme-corpus-balance/five-seeds-20261011-v1/results.md)、[監査結果](../../artifacts/phoneme-corpus-balance/five-seeds-20261011-v1/completion-audit.json)を参照。

### 校正移行と同じ他人受入率上限での比較

投入比率検証をPR #75としてマージ後、[保存済みスコアの診断](../phoneme-calibration-transfer/diagnose.py)を2026-10-11に完了した。新しい学習・録音は行わず、同じ5seed・5条件・3コーパスを使った。

事前にFAR目標0.1/0.2/0.5/1.0%を固定した。校正群の採点可能な他人試行だけから全300閾値を決めて保存した後、testのFAR・本人拒否率を評価した。これとは別に、testの全他人入力を分母とするFAR上限0.1/0.2/0.5/1.0%で、同じ上限における本人拒否率を比較した。後者はtestラベルを使う事後ROC診断であり、実運用で選べる閾値や未知データの性能ではない。同点スコアは分割せず、上限を超えない離散的な点を使う。

**CV重点条件の改善は、他人受入を揃えても一部残った。** FAR上限0.2%での本人拒否率の5seed平均は次のとおり。実際のFARは全条件・全seedで、JVS0.200%、CV0.199%、JTube0.193%と各コーパス内で一致した。

| 評価コーパス | 元の280 | 均等280 | CV重点280 |
| --- | ---: | ---: | ---: |
| JVS 本人拒否率 | 6.320% | 8.507% | 11.333% |
| CV 本人拒否率 | 51.422% | 50.911% | 48.556% |
| JTube 本人拒否率 | 46.067% | 44.533% | 45.467% |

CV重点条件はCV本人拒否が5/5seedで低下し、差は平均−2.867ポイント（各seed −3.333〜−2.222ポイント）。FAR上限0.1%でも5/5seedで低下し、平均−4.267ポイントだった。一方、上限0.5%では改善幅が平均−0.711ポイント・4/5seed、上限1.0%では平均＋0.267ポイント・4/5seed悪化となる。低FAR域の改善であり、ROC全体で一律に優れているわけではない。

従来の校正FAR1%条件で見えたCVの本人拒否低下は−8.111ポイントだったが、同じ実測FAR約0.2%での差は−2.867ポイントに縮まった。比較の閾値が異なるため厳密な因果分解ではないが、前の大きな改善幅をそのまま分離性能の向上と解釈できない。均等条件のCV改善は同上限で平均−0.511ポイント・3/5seedにとどまる。

**JVSの悪化は閾値の移行だけでは説明できない。** CV重点条件は同FAR上限0.2%でも本人拒否が平均＋5.013ポイント、5/5seedで悪化した。0.1/0.5/1.0%の全上限でも5/5seed悪化し、均等条件も同様に全上限・全seedで悪化した。JTubeでは均等条件が全4上限・5/5seedで本人拒否を低下させた。CV重点条件のJTubeは上限0.2%で平均−0.600ポイント・4/5seed低下と小幅で、上限1.0%ではほぼ同等となった。

校正からtestへのずれも確認した。校正で採点可能な他人のFAR約1%を狙ったとき、testの採点可能分のFAR平均はCVで元の280の0.307%に対しCV重点0.951%、JVSでは0.706%に対し1.771%だった。全入力のFARと分母が異なることに注意する。このずれと同FAR診断の両方から、CV重点は低FAR域の候補ではあるが、全体の置換モデルとしては採用しない。現在のモデル・運用閾値は更新していない。

次の優先は、学習比率の追加探索より、CV/JTubeで35〜40%残っている未採点の原因内訳を調べること。今回も採点率は変わらず、本人拒否率の大きな下限となっている。既存入力から音素不足・前処理失敗・登録側不足を分け、追加録音なしで改善余地を確認する。

新規テスト4件（同点、未採点分母、端点、校正のtest非参照）とRuff検査を通過した。[監査](../phoneme-calibration-transfer/audit.py)では900判定点（校正300・固定閾値test300・事後ROC300）を再集計し、ROC上限と閾値直下の違反、平均/SD・seed内差を確認した。元の校正FAR1%の75結果も完全再現した。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-calibration-transfer/tests
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-calibration-transfer/diagnose.py execute
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-calibration-transfer/audit.py
```

全4目標/上限の結果は[集計JSON](../../artifacts/phoneme-calibration-transfer/fixed-grid-20261011-v1/results.json)、[結果表](../../artifacts/phoneme-calibration-transfer/fixed-grid-20261011-v1/results.md)、[監査結果](../../artifacts/phoneme-calibration-transfer/fixed-grid-20261011-v1/completion-audit.json)に保存した。全testは既観測で、5seedのSDは独立の人物母集団の信頼区間ではない。

詳細はローカルの[全指標](../../artifacts/phoneme-corpus-scaling/comparison-20261011-v1/results.json)と[結果表](../../artifacts/phoneme-corpus-scaling/comparison-20261011-v1/results.md)。入力・モデル・閾値hashは同runの各freezeに保存した。追加データのQC後区間数はtrain312,690・validation48,326・test40,234。全9,000件の監査、各split200区間の特徴量再計算（差0）、既存音声・新規音声間の同一PCM検査（除外対象0件）を通過した。人物レベルの重複がないという意味ではない。

## 前処理停止からの復旧

5,370件目のASRテキストから音素を生成できず停止したため、[resume_pipeline.py](resume_pipeline.py)で同じ空音素エラーを再現して`g2p_unusable`として記録し、差し替えずに継続した。元の固定済み処理コードは変更せず、失敗状態・ASR・音声・復旧コードのhashを`recovery-events`と`recovery-policy.json`へ保存した。初回の復旧コードも同ディレクトリに保存している。未知の例外は自動で除外しない。

復旧テストにはpyopenjtalkのあるalignment venvを使う。全17テストとRuffが通過した。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-corpus-scaling/tests
```

## 候補の確認

2026-10-11に配布元の現行情報と実ファイルを確認した。

| 候補 | 今回の判断 |
| --- | --- |
| [JTubeSpeech-ASV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jtubespeech-asv_corpus) | 第一候補。公式説明は1,792話者・484時間、日本語限定subsetは979話者・245時間。研究開発目的に限定される暫定条件で、今回の研究用入力として扱う。商用モデルへの採用判断は別途必要 |
| SRC4VC | 既存の[聴取所見](../../docs/training-data-adoption.md)で切り出し品質に懸念があり採用保留。今回、未解決のまま再投入しない |
| [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) | 規模だけで選ばない。ASR用の音声・テキストをそのまま使っても、話者分類に必要な一貫した人物ラベルの問題は解決しない |
| [ITAコーパス Rion版](https://github.com/Rion-Dev/ita-corpus-Rion) | 検索結果には50話者・CC BY表記が残るが、配布元は2026-09-28から利用条件確認のため新規配布停止と告知。今回の取得候補から外す |
| [J-MAC](https://www.nii.ac.jp/dsc/idr/speech/submit/J-MAC.html) | 音声は同梱されず別途購入が必要。現在の自動取得対象に含めない |

JTubeSpeech-ASV ver1.1のZIPを部分取得し、目次・README・日本語リスト・公式評価リストを確認した。書き起こしファイルは同梱されていない。したがって今回はローカルASRによる機械生成テキストを使う事前確認であり、人手の正解テキストと同等には扱わない。日本語リストにも外国語の発音例を含む教育音声などがありうる。

## 固定済みの取得範囲

日本語リストは46,542発話・979ラベルで、50発話以上あるのは184ラベルだった。これらは公式testの92ラベルとIDが重複しない。固定seedのhash順でtrain 140・validation 20・test 20ラベルへ分割し、各50発話、計9,000発話を選んだ。trainの先頭70ラベルを小さい追加条件とする。コーパス間の実在人物の重複は不明で、人物の再特定は行わない。

分割は新しい音声の採点・抽出前に`selection-freeze.json`へ保存した。train先頭3ラベル×各先頭4発話＝12発話を事前確認に使う。スコアのよい音声へ差し替えず、validation/testは採用前の品質探索に使わない。取得時は固定ETag・HTTP Range・ZIP CRC・ファイルSHA-256を確認する。全ZIPを取得したわけではなく、配布者が提供するSHA-256を検証したという意味でもない。

## 比較設計案

抽出後の話者・音素支持数を確認してから学習設計を最終固定する。取得の分割固定と、モデル学習条件の固定は別段階。

| 条件 | JVS | Common Voice | JTubeSpeech-ASV | 狙い |
| --- | ---: | ---: | ---: | --- |
| 現行構成の対照 | 70 | 70 | 0 | 140ラベルの基準 |
| 同数で第三コーパスを混ぜる対照 | 70 | 35 | 35 | 140ラベルのままデータ源を変える |
| 第三コーパス70追加 | 70 | 70 | 70 | 210ラベルへの拡大 |
| 第三コーパス140追加 | 70 | 70 | 140 | 280ラベルへの拡大 |

同じ学習更新数・初期化seed・特徴処理・損失・音素方針で比較し、最終checkpointを使う。まず同じseedで傾向を見て、複数seedによる再現確認を加える。学習音声量やラベル品質まで自動的に同一になる比較ではないので、話者数だけの因果効果とは表現しない。コーパス比率、各話者の発話数・音素数・学習投入数も併記する。新規条件は既存モデルへの継ぎ足しだけで比較せず、対照と同じ初期化から学習する。

評価はJVS・Common Voiceの既存分割に加え、第三コーパス内で学習から分離した20校正・20testラベルを使う。コーパスごとのEER、校正群だけで決めた閾値でのFAR・FRR、未採点を含む本人拒否、採点率を示し、平均値だけで改善としない。第三コーパスの評価も機械生成テキストに依存するため、その誤りの影響は別の制約として残る。既存の本人実録音は補助診断にとどめ、学習・閾値調整には使わない。

## 実行経路

`artifacts/phoneme-corpus-scaling/source/`に配布ZIPの目次、README、Japanese.lst、Original.lst、testリストを保存する。ASRは既存のencoder用venvと分け、モデルrevisionとファイルhash、パッケージ版、デコード設定を保存する。録音は外部ASRサービスへ送らない。

```sh
python3 poc/phoneme-corpus-scaling/inspect_source.py
python3 poc/phoneme-corpus-scaling/acquire.py select
python3 poc/phoneme-corpus-scaling/acquire.py pilot
HF_HOME="$PWD/artifacts/phoneme-corpus-scaling/hf-cache" artifacts/phoneme-corpus-scaling/asr-venv/bin/python poc/phoneme-corpus-scaling/transcribe.py --pilot
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/pilot_quality.py
python3 poc/phoneme-corpus-scaling/acquire.py download
```

ASRモデルは`mlx-community/whisper-large-v3-turbo`のrevision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`を使用した。ローカルモデルの実パス・取得revisionはGit管理外の`artifacts/phoneme-corpus-scaling/asr-model.json`で指定する。初期取得・環境準備は既存の研究環境で実施済み。モデルのファイルhashは`pilot-asr/freeze.json`に保存される。

`acquire.py`の分割は固定後に上書きしない。異なる条件では新しいrunを作る。`pilot_quality.py`は既存Runtimeを再利用するためメタデータのsplitが`pilot`となる。これらの事前確認出力をそのまま正式train/validation/test manifestへ流用しない。

新しい学習データは[既存の採用手順](../../docs/training-data-adoption.md)に従う。利用者の依頼により、追加学習の前に[ランダム100区間のレビュー](review.md)を用意した。その後、利用者から「確認した・全件を見るのは大変なので保存はしていない・検証を進めて」と指示されたため、個別回答の保存を要求せず通常の検証を再開した。回答数・品質率・正式な採用判定は自動生成しない。アライメント成功をラベル正解の証拠とせず、ASR由来の不確かさを明示する。現行モデルや凍結済みマイクパイロットは変更しない。


## 9,000発話の準備を再開

実行設定は[corpus-config.json](corpus-config.json)。runは`artifacts/phoneme-corpus-scaling/expanded-inputs-20261011-v1`。レビュー用run・ASR・回答は保持する。

```sh
# 初回だけ、元manifest・コード・条件を固定
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/corpus.py prepare
# 中断後も同じコマンドで完了分から再開
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/corpus.py pipeline
# 進捗
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/corpus.py status
# 完了分の監査。全件完了後は --partial を外す
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/audit_corpus.py --partial
```

書き起こしは分離したMLX環境で実行し、音素抽出は元の数値計算環境で並行して処理する。個々の発話の結果は一時ファイルからatomic renameで確定し、読取り途中のファイルを抽出側が使わない。プロセスの二重起動をロックで防ぐ。

レビューで使った30秒の発話長上限・Julius 10秒上限・30〜250ms・RMS −50dBFS以上を保持する。30秒超過はASR前に除外し、短く切り直したり別発話で補充したりしない。低音量などの除外区間・位置合わせの失敗・音素別支持数を保存する。音素品質のレビューで未確認だった `/by/`・`/dy/`・`/v/` を含め、全36音素の品質が確認済みとは扱わない。

正式キャッシュでは元のtrain/validation/test分割を復元し、Runtime内部の`pilot`は`extraction_split`へ残す。元MP3・ASR・WAV・音素区間・特徴量の対応をhashで追跡する。特徴正規化は既存のtrain限定statisticsを共通利用し、追加コーパスのvalidation/testで再推定しない。

この準備プロセスはモデルを学習・更新しない。9,000発話の完了と監査後、各話者・音素の実際の支持数を確認して4条件の学習manifestと更新数を固定し、比較学習へ進む。未完了の入力で学習を始めない。校正群だけで閾値を決め、testを見た音素選択・checkpoint選択はしない。本人の追加録音は不要。

## 4条件の学習・評価を自動実行

利用者から学習への進行が指示されたため、[study-config.json](study-config.json)で4条件・学習予算・評価規則を固定した。runは`artifacts/phoneme-corpus-scaling/comparison-20261011-v1`。`study.py execute --wait-for-corpus`が準備完了を待ち、全件監査、入力確定、3モデルの新規学習、校正、test評価まで進む。監査・支持数の不足・プロセス停止を検出した場合は失敗を記録し、不完全な条件で学習・評価を続行しない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/study.py freeze
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/study.py execute --wait-for-corpus
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/study.py status
```

- 対照140話者は完了済みall36を再利用する。encoder・seed・更新数・正規化と元の入力hashを照合する。
- 置換140話者はJVS70・CV35・JTube35、拡張はJVS70・CV70にJTube70または140を追加する。CV35は固定seedのhash順、JTubeは元の固定分割順の先頭から選ぶ。学習用の音素支持数で全36音素を維持できることを確認する。
- 新規3モデルは初期encoderを共通化し、各216,000更新・warmup1,000・最終checkpointを使う。分類headのクラス数は140/210/280に対応する。各音素のgroup数と最大投入区間数は揃えるが、希少音素の実投入数と各コーパスの音声量は異なりうる。
- `scaling_learner.py`は既存の`phoneme-greedy-selection/learner.py`と同じ学習計算を使い、140固定だったcohort検証・記録を実際の話者数へ拡張した。元の凍結コードは変更しない。
- 音素抽出環境はtorch 2.14.0、元の学習環境はtorch 2.14.1だったため、学習は既存モデルと同じverification venvで実行する。最初の200区間の特徴量再計算は差0。全件準備後も各split200区間を固定seedで再計算し、最大絶対差2e-6以内を確認してから学習する。これは全区間の再計算検証ではない。
- 新規音声のPCM WAVが既存の学習・評価音声と一致する場合、または複数の新規MP3が同一PCM WAVを持つ場合は、その新規音声の全区間を一律除外する。query枠は未採点として残し、別音声で補充しない。
- 評価はJVSの既存入力、CVの既存60ラベルholdout、JTubeの新しい20校正＋20testラベルを使う。JTubeは固定50発話の先頭20を登録、残り30をqueryとする。JVSの登録区間は既存条件、CV・JTubeは音素あたり最大30区間。登録量までコーパス間で同一とする比較ではない。
- 5母音を必須とし、利用可能な共通追加音素を等重みで統合する。閾値はモデル・コーパスごとに校正群のFAR1%から決め、全モデルと全閾値を固定してからtestを採点する。
- EERは採点可能分、FAR・FRRは未採点を拒否として含む全入力を報告する。FRR・FARの基準との差にはquery話者単位のpaired bootstrap 1,000回の区間を付ける。基準モデルの校正スコア18,000件が既存実装と完全一致することを確認済み。

今回の実行は1 seedの改善傾向確認。JVS・CVのtestは既観測であり、未観測testとして扱わない。実在人物のコーパス間重複、ASRラベル品質、音声量・録音環境の違いが残るため、純粋な話者数の因果効果や運用性能の証明とはしない。結果はrun配下の`results.json`・`results.md`へ自動保存する。学習完了前に改善の有無を断定しない。
