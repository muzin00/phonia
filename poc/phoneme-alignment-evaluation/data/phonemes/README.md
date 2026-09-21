# Expected phonemes

`expected.jsonl`には、サンプル音声の発話テキストをpyopenjtalkで変換した読みと生の音素列を記録します。

生成方法:

```bash
cd poc/phoneme-alignment-evaluation
uv sync
uv run python scripts/generate_phonemes.py
```

`raw_phoneme_sequence`と`raw_phonemes`はpyopenjtalkの出力を改変せず保存します。使用したpyopenjtalkと同梱辞書のバージョンも各レコードに記録します。`review_status`が`pending`のレコードは、人間による読みと音素列の確認が完了していません。

確認後の共通音素表記や方式別マッピングは、生の出力を上書きせず別フィールドまたは別ファイルへ記録します。
