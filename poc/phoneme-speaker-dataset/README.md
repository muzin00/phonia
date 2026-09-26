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
