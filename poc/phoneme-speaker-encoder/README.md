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

[Issue #21](https://github.com/muzin00/phonia/issues/21)で学習・評価仕様を設計した。
入力方式は事前に一つへ固定せず、可変長log-Mel、生波形などの候補を実際に学習し、
未知話者のvalidation照合性能から選択する。

入力データ、候補となる入力特徴、可変長処理、encoder、損失関数、batch sampling、成果物形式、
validationとtestの用途を設計済みである。判断が難しい項目は互換性のある候補を組み合わせ、
未知話者のvalidation照合性能から採用する。設計version 2.0.0では、本比較はlog-Mel 8通り・
生波形8通りの計16通りを維持し、規模を揃えた時間混合なしモデルと長文脈生波形モデルを
各1設定の限定比較として追加する。全18設定が採用対象である。

過学習確認6、成立性確認18、70話者比較54、採用構成の学習曲線追加9の計87学習runを計画する。
品質層・境界ずれ・音量への診断は再学習せず行う。採用規則を事前固定し、選択した構成の
3 seedすべてを凍結してtestを一度評価する。固定した128次元等の最適性や、入力方式全般の
優劣を保証する設計ではない。既存全発話方式との比較はPhase 7、実環境は外部評価段階で扱う。

Dataset / DataLoader、特徴統計、固定enrollment / trial生成を`phase3_data/`に実装した。
6種のencoder、AAM-Softmaxと母音内SupCon、optimizer境界checkpoint、固定validation照合を
`phase3_train/`に実装した。候補比較・test最終評価はこの実装とは別段階で行う。

## 4. ドキュメント

- [入力設計](input-design.md)
- [生波形encoder設計](waveform-encoder-design.md)
- [学習・評価設計](training-evaluation-design.md)
- [候補比較計画](comparison-plan.md)
- [70話者・30分上限pilot結果](phase3-70spk-pilot-results.md)
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
