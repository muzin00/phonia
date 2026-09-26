# 音素別話者データセット

## 1. 目的

Phase 2では、Phase 1で採用したJuliusのアライメント結果を、音素別話者表現の
学習と本人照合評価に利用できるデータセットへ整理する。

個別区間の人手選別や境界修正は行わず、機械的に学習入力を構成できる全区間を
同じ規則で使用する。元音声から各成果物までの由来を追跡可能にし、学習・検証・
評価間のデータリークを防ぐ。

## 2. 対象

- 日本語の5母音 `/a/ /i/ /u/ /e/ /o/`
- Juliusが出力した母音phone区間
- 学習、検証、評価に使用する話者と発話
- 元音声、音素区間、話者、収録条件、生成設定を対応付けるmanifest
- データセットの構成と品質を確認する機械集計

## 3. ドキュメント

- [コーパスの選定](corpus-selection.md)
- [JVSデータ調査](data-investigation.md)
- [学習データの品質項目](data-quality.md)
- [段階的な検証戦略](validation-strategy.md)
- [JVSデータセット設計](dataset-design.md)
- [母音区間データセットのスキーマと配置](dataset-schema.md)
- [母音区間データセットの生成・品質確認結果](dataset-results.md)
- [Phase 2母音区間の層化レビュー結果](review-results.md)

## 4. 母音区間生成

生成物はGit管理外の`data/generated/`へ出力する。まず発話manifestから期待音素列を
生成し、次にJuliusを発話単位で実行する。

```shell
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-speaker-dataset/scripts/generate_expected_phonemes.py

python poc/phoneme-speaker-dataset/scripts/run_julius_dataset.py

python poc/phoneme-speaker-dataset/scripts/build_vowel_manifest.py
```

Julius処理は成功済みの発話別JSONを検証して再利用するため、中断後に同じコマンドを
再実行できる。失敗は`data/generated/alignments/manifest.jsonl`に理由付きで記録し、
後続発話の処理を継続する。全量実行前の確認には各コマンドの`--limit`を使用できる。
母音区間manifestは元WAVとframe範囲を参照し、区間WAVを大量に複製しない。
全量処理の件数、区間長、品質フラグ、失敗理由は
[`data/vowel-dataset-validation.json`](data/vowel-dataset-validation.json)へ記録する。

## 5. 層化レビュー用データ

Issue #17のレビュー対象は、固定seedで5母音、データ分割、品質グループ、区間長を
層化し、音素変換を含む発話を追加して500区間抽出する。

```shell
python poc/phoneme-speaker-dataset/scripts/sample_review_segments.py

uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/build_review_dataset.py \
  --candidate julius=poc/phoneme-speaker-dataset/data/reviews/stratified/sample.jsonl \
  --dataset-id jvs-phase-2-stratified-vowel-review \
  --dataset-version 1 \
  --output poc/phoneme-speaker-dataset/data/reviews/stratified/review-dataset.json \
  --mapping-output poc/phoneme-speaker-dataset/data/reviews/stratified/candidate-map.json
```

レビューUIはJVSコーパスをメディアルートとして起動する。

```shell
cd tools/phoneme-review-ui
PHONIA_REVIEW_MEDIA_ROOT=../../poc/phoneme-speaker-dataset/data/source/jvs_ver1 \
PHONIA_REVIEW_DATASET=../../poc/phoneme-speaker-dataset/data/reviews/stratified/review-dataset.json \
PHONIA_REVIEW_OUTPUT=../../poc/phoneme-speaker-dataset/data/reviews/stratified/review-records.jsonl \
pnpm dev
```

保存した回答の条件別集計は次のコマンドで再生成する。

```shell
python poc/phoneme-speaker-dataset/scripts/summarize_review_results.py
```

## 6. Phase 3入力manifest

層化レビューで系統的な利用困難が確認された`near_silent`を、固定した機械規則で
全区間から一律に除外する。元のPhase 2 manifestは変更せず、Phase 3用の派生manifestを
生成する。

```shell
python poc/phoneme-speaker-dataset/scripts/build_phase3_manifest.py
```

- 正本設定: `config/phase3-input.json`
- Phase 3入力: `data/generated/phase3-vowel-segments.jsonl`
- 検証結果: `data/phase3-vowel-dataset-validation.json`

生成処理は元音声のメタデータとframe範囲、話者分割、学習曲線cohort、母音・評価用途の
カバレッジを検査する。Phase 3の学習処理は元の`vowel-segments.jsonl`ではなく、
`phase3-vowel-segments.jsonl`を入力の正本として使用する。
