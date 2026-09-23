# 40-utterance expansion set

パイロットで確認した評価手順を30〜50発話へ拡張するためのデータ領域です。

`scripts/prepare_expansion_samples.py` は
[`MontrealCorpusTools/japanese-jvs-demo`](https://github.com/MontrealCorpusTools/japanese-jvs-demo)
の固定リビジョンから、`jvs001`〜`jvs004`ごとに、WAVとラベルがそろう発話を
ファイル名順で10件ずつ選び、合計40発話を取得します。人手による品質選別や信頼度の
付与は行いません。

```bash
uv run python scripts/prepare_expansion_samples.py
```

JVSの利用条件に従い音声とラベル本体はGit管理せず、生成される`manifest.jsonl`に
出典リビジョン、音声メタデータ、SHA-256を保存します。評価を実行する環境ごとに上記
コマンドで音声を取得してください。

## Results

- 発話数: 40（4話者 × 10発話）
- 音声長: 224.04秒
- 期待母音単位: 1,164
- Julius: 40/40発話で期待列一致、1.008秒
- Wav2Vec2: 40/40発話で期待列一致、11.960秒
- MFA: 40発話を処理、辞書・G2Pの読みが共通期待列と一致したのは12発話、88.693秒

集計結果は[`evaluation-summary.json`](evaluation-summary.json)に保存しています。MFAの
不一致は、テキストから自身の辞書・G2Pを通す評価経路とpyopenjtalk由来の共通期待列の
読みが異なる発話です。アライメント処理そのものの失敗とは区別しています。

入力音声を取得した後、期待列の生成から集計までを次の順序で再実行できます。各方式の
環境構築はプロジェクト直下READMEからリンクした方式別READMEを参照してください。

```bash
uv run python scripts/generate_phonemes.py \
  --input data/expansion/manifest.jsonl \
  --output data/expansion/expected.jsonl
uv run python scripts/generate_vowel_units.py \
  --input data/expansion/expected.jsonl \
  --output data/expansion/expected-vowels.jsonl

# 各方式のrun・normalizeと、extract_vowel_segments.pyによる機械検査を実行後
uv run python scripts/summarize_expansion_evaluation.py
```
