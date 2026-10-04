# 音素別話者encoder

## 1. 目的

Phase 3では、母音の短い波形区間から固定長の話者embeddingを生成する共通encoderを
学習し、学習に含まれない話者の登録音声と照合音声でも話者差を捉えられるか検証する。

同じ母音という条件の中で、次の関係を学習目標とする。

```text
同じ話者・同じ母音   → embedding空間で近づける
異なる話者・同じ母音 → embedding空間で遠ざける
```

## 2. 対象範囲

対象は、日本語の5母音 `/a/ /i/ /u/ /e/ /o/` の1区間を入力とする共通encoderである。
共通encoderはtrain話者で学習し、validation、test、新規ユーザーの登録時には更新しない。

複数母音のembeddingやスコアをTransformerで統合する処理は、
[Issue #22](https://github.com/muzin00/phonia/issues/22)で扱う。

## 3. 現在の状態

[Issue #21](https://github.com/muzin00/phonia/issues/21)で学習・評価仕様を設計し、
可変長log-Melと生波形を含む18設定を未知話者のvalidation照合性能で比較した。

入力データ、候補となる入力特徴、可変長処理、encoder、損失関数、batch sampling、成果物形式、
validationとtestの用途を設計済みである。判断が難しい項目は互換性のある候補を組み合わせ、
未知話者のvalidation照合性能で比較した。設計version 2.0.0では、本比較はlog-Mel 8通り・
生波形8通りの計16通りを維持し、規模を揃えた時間混合なしモデルと長文脈生波形モデルを
各1設定の限定比較として追加した。全18設定を採用対象とした。

過学習確認6、成立性確認18、70話者比較54を完了した。18設定×3 seedの54 runに失敗・未実行はなく、
採用構成は`log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel`に決定した。
validationの主条件における5母音平均EERは3 seed平均で9.436%だった。
採用構成の学習曲線追加9 runと、品質層・境界ずれ・音量へのvalidation診断も完了した。
結果と制約は[Phase 3学習曲線・感度診断](phase3-learning-curve-diagnostics-results.md)に記録した。
採用規則と3 seedのcheckpoint・閾値・手順を固定して最終testを一度評価し、
Phase 3の計画済み学習・診断・最終評価は完了した。testの平均macro EERは10.028%、
validationで固定したFAR 1%用閾値でのtest平均FAR/FRRは1.068%/43.294%。
結果は[Phase 3最終test評価](phase3-final-test-results.md)に記録した。
固定した128次元等の最適性や、入力方式全般の優劣、実運用で認証に十分な性能を
保証する設計ではない。既存全発話方式との比較はPhase 7、実環境は外部評価段階で扱う。

Dataset / DataLoader、特徴統計、固定enrollment / trial生成を`phase3_data/`に実装した。
6種のencoder、AAM-Softmaxと母音内SupCon、optimizer境界checkpoint、固定validation照合を
`phase3_train/`に実装した。候補比較とtest最終評価は完了した。

Phase 4の固定encoderによる登録・保存・読込は`../phoneme-user-registration/`に実装した。
初期仕様は全5母音・各10区間以上。入力条件、プロファイル形式、Phase 5の利用境界と
実行方法は[ユーザー登録設計](../phoneme-user-registration/README.md)を参照する。

## 4. ドキュメント

- [入力設計](input-design.md)
- [生波形encoder設計](waveform-encoder-design.md)
- [学習・評価設計](training-evaluation-design.md)
- [候補比較計画](comparison-plan.md)
- [70話者・30分上限pilot結果](phase3-70spk-pilot-results.md)
- [70話者・3 seed比較結果と採用構成](phase3-70spk-comparison-results.md)
- [採用構成の学習曲線・validation感度診断結果](phase3-learning-curve-diagnostics-results.md)
- [Phase 3最終test評価結果](phase3-final-test-results.md)
- [Phase 4ユーザー登録設計と実行方法](../phoneme-user-registration/README.md)
- [参照log-Mel設定](config/baseline-log-mel.json)
- [log-Mel encoder設定](config/log-mel-encoders.json)
- [探索空間](config/search-space.json)
- [共通実験プロトコル](config/experiment-protocol.json)
- [生波形encoder設定](config/waveform-encoders.json)
- [研究全体の設計](../../docs/research-design.md)
- [Phase 2音素別話者データセット](../phoneme-speaker-dataset/README.md)

## 5. 設計設定の整合性検証

Phase 3データ入力のテストと実行にはPython 3.12、NumPy、PyTorchが必要である。
依存関係は`pyproject.toml`に記録した。`uv sync --project poc/phoneme-speaker-encoder`
で環境を作成できる。

```sh
uv run --project poc/phoneme-speaker-encoder \
  python -m unittest discover -s poc/phoneme-speaker-encoder/tests -v
```

## 6. データ入力と成果物

`phase3_data.input.SegmentDataset`はmanifestの`source_file`と`start_frame:end_frame`から
元WAVを読み、`InputPipeline`でcrop、DC除去、任意のRMS正規化、log-Melを適用する。
生波形には`kind="waveform"`を使う。`collate_segments`は右paddingと有効長maskを返す。
学習時は`BalancedSampler`、`phase3_data.sampling.VowelMicrobatchSampler`を
DataLoaderの`batch_sampler`へ渡し、Datasetを`mode="random"`とrun seedで作る。
再開時は、prefetchで先読みしたsampler位置を使わず、実際に完了したupdate数を
`state_at(completed_updates)`へ渡して保存する。読み込み時は`load_state_dict`で検証する。

以下のCLIは成果物をGit管理外の`artifacts/`へ出力する。統計はcohortとRMS条件ごとに
8回実行する。標準出力に件数とファイルSHA-256を表示し、`data.json`にmanifest、cohort、
前処理、特徴統計、計算コードのchecksumを保存する。実データ全件の統計生成には時間がかかる。

```sh
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/build_data_artifacts.py statistics \
  --cohort 10 --rms on \
  --output artifacts/phoneme-speaker-encoder/cohort-10-rms-on/feature-statistics.json
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/build_data_artifacts.py selections \
  --split validation --output artifacts/phoneme-speaker-encoder/fixed-validation
```

`selections`は`selections/enrollment-segments.jsonl`と`selections/trials.jsonl`、
それらのSHA-256を含む`data.json`を生成する。test用は設定を凍結した最終評価段階で
`--split test`として生成する。統計の読み込みには`load_feature_statistics`を使い、
manifest・cohort・前処理のchecksumと統計自身の内容checksumを検証する。

## 7. 学習とvalidation評価

`scripts/run_phase3.py`は`statistics_mlp`、`tdnn`、`framewise_cnn`、
`waveform_cnn_k80`、`waveform_cnn_k240`、`waveform_cnn_k240_context27`を扱う。
通常の学習は5母音×20件を1 logical updateとして勾配を蓄積し、指定間隔で固定validationを
全件評価する。`--supcon off`、`--rms off`も選択できる。log-Melでは対応するcohort/RMSの
特徴統計が必須で、生波形では`--statistics`は指定しない。

```sh
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/run_phase3.py train \
  --encoder tdnn --cohort 10 --rms on \
  --statistics artifacts/phoneme-speaker-encoder/cohort-10-rms-on/feature-statistics.json \
  --maximum-updates 2000 --warmup-updates 100 --validation-interval 250 \
  --output artifacts/phoneme-speaker-encoder/sanity-tdnn-seed20260926
```

`--resume`は同じoutputの`checkpoints/last.pt`から再開し、入力・設定・コードのchecksumを
検証する。固定32区間の過学習確認には`train`の代わりに`overfit`を使う。`train`の出力には
`run.json`、`training/history.jsonl`、`training/summary.json`、`checkpoints/{last,best}.pt`、
各評価updateの`validation/update-*/{scores,metrics,selections}`が含まれる。
検証を個別に再実行する場合は次の通り。

```sh
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/run_phase3.py evaluate \
  --run-dir artifacts/phoneme-speaker-encoder/sanity-tdnn-seed20260926
```

小規模の動作確認では`train --skip-validation --target-update 1 --evaluate
--max-eval-queries-per-vowel 1`を指定できる。この場合の`partial: true`の指標や閾値は
性能比較・閾値採用に使用しない。test splitの評価は採用設定の凍結前には行わない。

## 8. 10話者・18設定の成立性確認

`config/search-space.json`の本比較16設定・限定比較2設定を
`scripts/run_sanity_matrix.py`で展開する。`config/execution-budget.json`は実行前に
固定したCPU・float32・0 worker・時間/メモリ/容量上限で、探索途中の性能を見て変更しない。

```sh
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/run_sanity_matrix.py plan
uv run --project poc/phoneme-speaker-encoder \
  python poc/phoneme-speaker-encoder/scripts/run_sanity_matrix.py run
```

実行は逐次で、`artifacts/phoneme-speaker-encoder/comparisons/phase3-sanity-v2-10spk-seed20260926/`
に固定matrix、予算、各runの設定・環境・データchecksum・学習履歴・checkpoint・固定validation
trialの複製・score/指標・再評価照合結果・資源使用量を保存する。最初のrunは250 updateでcheckpointを作成し、
そこから再開する。`report`コマンドは既存runの状態を再集計し、未着手・失敗も一覧に残す。
中断後に`run`を再実行すると完了済みrunは飛ばし、実行途中のcheckpointから再開する。
この10話者EERは成立性診断であり、構成の採否や順位決定に使わない。testは使用しない。

## 9. 70話者・3 seedの本比較

`config/full-execution-budget.json`でCPU 4並列、1 run最大16時間、空き容量下限400 GiBを
事前固定する。`scripts/run_full_matrix.py`は18設定×3 seedの54 runを凍結し、実測時間だけを
実行順に用いる。10話者・短時間pilotのEERによる候補除外は行わない。

```sh
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/run_full_matrix.py plan
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/run_full_matrix.py run
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/run_full_matrix.py report
```

各runの採用checkpointに対応するscore・曲線は完全保存し、採用されなかった途中評価の
score・曲線だけをchecksum付きの小さい指標へ整理する。これにより保存容量を抑えつつ、
各評価時点のmacro EERと最終採用根拠を残す。選択checkpointの再評価一致も検証する。
`run`は既存の完了runを飛ばし、中断したrunはoptimizer境界checkpointから再開する。
test splitは使用しない。

### ブラウザで比較結果を見る

完了した全54 runの選択checkpointのvalidation指標から18設定の順位表を生成できる。
3 seed平均macro EERの順位、seed間標準偏差、母音別・cross-text・区間長別評価、
学習時間と観測RSSを表示する。共通の推論benchmarkも完了し、採用構成を記録済みである。

```sh
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/build_comparison_report.py
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2/browser-report
```

`http://127.0.0.1:8765/`を開く。設定検索・指標別の並び替え・行選択での詳細表示・CSV保存に対応する。
生成先の`index.html`はデータを埋め込んだ単独HTMLとして直接開くこともできる。
`ranking.json`に集計値と実行計画・結果のSHA-256を保存する。生成物はGit管理外の成果物に置く。

### 保存済みcheckpointの追加評価と採用記録

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/evaluate_comparison.py run
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/build_comparison_report.py
```

validation主trialの対応付き話者bootstrapを10,000回実行し、54 checkpointを
共通1,000 query・batch=1で専用CPUプロセスごとにbenchmarkする。既存の完了キャッシュは
checksumを検証して再利用する。`selection-evaluation/`に抽出話者ID、seed別replicate、
benchmark、順位・採用記録と3 seedのcheckpoint/閾値bundleを保存する。
ブラウザページに差の95%区間・推論時間・計測RSS・採用記録を追加し、`browser-report/results.md`も生成する。

CPU benchmark API・スレッド数は元の実行予算に未記載だったため、今回の追加計測前に
`selection-evaluation/evaluation-budget.json`で補完した。結果確認後の補完であること、
RSSは5 ms間隔のサンプリング最大値であることを結果文書に記載している。testは使用しない。

## 10. 凍結した最終test評価

採用済み3 checkpointと各seed自身のvalidation閾値を固定し、testの15話者で
登録・照合を一度実行した。test内で閾値を再較正せず、候補やseedも選び直していない。
評価コード、seed別・品質層別の結果、成果物checksumは
[最終test評価結果](phase3-final-test-results.md)を参照する。
