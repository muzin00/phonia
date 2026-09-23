# Extracted vowel segments

アライナーが出力した時間境界を使い、元音声から母音区間を再生成する。

ここでいう母音区間は、対象母音の音響的な実現を含み、別の母音の核を含まない連続区間である。前後の子音との遷移や共調音が境界付近に残ることは許容し、音声学的に純粋な母音定常部だけを意味しない。

## MFA

リポジトリルートから次を実行する。

```shell
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/extract_vowel_segments.py
```

`mfa/`以下へ次の成果物を生成する。

- `exact/`: MFA境界どおりの母音WAV。子音との遷移を含む場合がある。Git管理しない
- `context/`: exact区間の前後100msを含むレビュー用WAV。Git管理しない
- `manifest.jsonl`: 出力元、サンプル単位の境界、RMS、チェックサム、品質フラグ
- `validation.json`: 生成数と品質フラグの集計

境界秒は元音声のサンプル位置へ丸め、実際に切り出した境界もmanifestへ記録する。元音声は変更しない。

MFAの生出力には子音を含む全音素の境界を保持する。必要になった場合はそこからモーラ区間を再構成できるため、この段階ではモーラWAVを生成しない。
