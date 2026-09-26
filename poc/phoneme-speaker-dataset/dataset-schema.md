# 母音区間データセットのスキーマと配置

## 1. バージョン

初期スキーマを`schema_version = 1`、設計版を`design_version = 1.0.0`とする。
正本は[`config/vowel-dataset.json`](config/vowel-dataset.json)である。生成時には設定、入力
manifest、期待音素列、Julius実行ファイル、音響モデルのSHA-256をrun metadataへ記録する。

スキーマのフィールドを削除する、意味を変更する、必須性を変更する場合は
`schema_version`を上げる。閾値や配置規則など、互換性を保った設計変更では
`design_version`を上げる。

## 2. 配置

生成物は`data/generated/`へ置き、元音声と同様にGit管理外とする。

```text
data/generated/
├── expected-phonemes.jsonl
├── expected-phonemes-failures.jsonl
├── expected-phonemes-run.json
├── alignments/
│   ├── manifest.jsonl
│   ├── run.json
│   ├── raw/<speaker_id>/<utterance_id>.json
│   └── failures/<speaker_id>/<utterance_id>.log
├── vowel-segments.jsonl
└── failures.jsonl
```

全量manifestから再生成できるコンパクトな集計結果は
`data/vowel-dataset-validation.json`としてGit管理する。

ファイル名には元WAVのstemではなく`utterance_id`を使用する。JVSの`parallel100`では
異なる話者が同じ元WAV名を持つため、stemだけを使用してはならない。

## 3. 期待音素列

`expected-phonemes.jsonl`は発話ごとに一件とし、次を必須とする。

- `schema_version`, `utterance_id`, `speaker_id`
- `source_file`, `source_sha256`, `transcript`
- `split`, `evaluation_role`, `learning_curve_cohorts`, `session_id`
- `g2p_engine`, `g2p_version`, `g2p_dictionary`
- `reading_kana`, `raw_phonemes`

G2Pに失敗した発話は成功manifestへ含めず、同じ`utterance_id`と`stage`、`reason`、
`error_type`を`expected-phonemes-failures.jsonl`へ記録する。

## 4. アライメントmanifest

`alignments/manifest.jsonl`は入力発話ごとに一件を持ち、`status`を`success`または
`failure`とする。共通フィールドは次のとおりである。

- 発話の由来: `utterance_id`, `speaker_id`, `source_file`, `source_sha256`
- データ用途: `split`, `evaluation_role`, `learning_curve_cohorts`, `session_id`
- 実行条件: `aligner`, `aligner_version`, `model_id`, `model_version`
- 再開判定: `alignment_input_sha256`

成功時は`raw_alignment_file`と`phone_interval_count`を持つ。失敗時は`stage`,
`error_type`, `reason`と、存在する場合は`log_file`を持つ。処理は発話単位で継続し、
ある発話の失敗によって後続発話を停止しない。

pyopenjtalkの`v`と`ty`は固定したJulius単音素モデルに存在しない。Julius付属の
`yomi2voca`がヴ行を/b/へ変換する規則に合わせて`v`を`b`へ、口蓋化した/t/はモデルに
存在する最も近い`ch`へ変換する。規則は設定の`phone_mapping_version`と
`unsupported_phone_mapping`へ記録し、変換後の列もチェックポイント指紋に含める。

## 5. 母音区間manifest

`vowel-segments.jsonl`はJuliusが生成した原子的な母音phone区間ごとに一件とし、
少なくとも次を保持する。

- 由来: 発話、話者、元音声、元音声チェックサム、元音素位置
- 用途: split、評価用途、学習曲線cohort、セッション
- 音素: 正規化母音、元音素、前後音素、無声化、長母音属性
- 境界: Julius境界、元音声上で丸めたframe境界、実際の開始・終了時刻
- 生成条件: Julius、音響モデル、G2P、スキーマ、設計の各バージョン
- 波形: 元WAVのパスとSHA-256、読み出すframe範囲、形式、RMS
- 分析属性: 短区間、無音、低レベルなどの`quality_flags`

品質フラグは分析用であり、採否、重み付け、サンプリングには使用しない。時刻不正、
音声範囲外、読み出し不能、期待音素列との不一致など、学習入力を構成できない場合だけを
失敗として`failures.jsonl`へ記録する。

初期版は`storage_mode = source_slice`とし、区間ごとのWAVを複製しない。全量生成で約47万区間となるため、
個別ファイルにするとファイル数が過大になるため、Phase 3は`source_file`を開き、
`start_frame`以上`end_frame`未満を読み出す。構築時にも同じ読み出しを実行してRMSを計算し、
各区間が実際に学習入力を構成できることを検査する。将来shard形式へ変換する場合も、この
manifestを正本とする。

## 6. バージョン管理

Gitでは設定、生成プログラム、テスト、スキーマ文書、集計結果を管理する。JVS音声、
Juliusの中間入力、発話別アライメント、大規模manifestは管理しない。
成果物の同一性はrun metadataに記録した入力と設定のチェックサムで確認する。
