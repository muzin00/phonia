# Extracted vowel segments

アライナーが出力した時間境界を使い、元音声から母音区間を再生成する。

## MFA

リポジトリルートから次を実行する。

```shell
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/extract_vowel_segments.py
```

`mfa/`以下へ次の成果物を生成する。

- `exact/`: MFA境界どおりの母音WAV。Git管理しない
- `context/`: exact区間の前後100msを含むレビュー用WAV。Git管理しない
- `manifest.jsonl`: 出力元、サンプル単位の境界、RMS、チェックサム、品質フラグ
- `validation.json`: 生成数と品質フラグの集計

境界秒は元音声のサンプル位置へ丸め、実際に切り出した境界もmanifestへ記録する。元音声は変更しない。
