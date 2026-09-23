# Expected phonemes

`expected.jsonl`には、サンプル音声の発話テキストをpyopenjtalkで変換した読みと生の音素列を記録します。

生成方法:

```bash
cd poc/phoneme-alignment-evaluation
uv sync
uv run python scripts/generate_phonemes.py
uv run python scripts/generate_vowel_units.py
```

`raw_phoneme_sequence`と`raw_phonemes`はpyopenjtalkの出力を改変せず保存します。使用したpyopenjtalkと同梱辞書のバージョンも各レコードに記録します。`review_status`が`pending`のレコードは、人間による読みと音素列の確認が完了していません。

確認後の共通音素表記や方式別マッピングは、生の出力を上書きせず別フィールドまたは別ファイルへ記録します。

`expected-vowels.jsonl`には、`expected.jsonl`の生音素列から抽出したモデル非依存の期待母音単位を記録します。G2Pが出力した母音phoneごとに一件を作り、隣接する同一母音も結合しません。長母音記号を含む一つのphoneは一件のまま保持し、`expected_units`で展開後の母音数を表します。

- `vowel_index`: 発話内で0から始まる母音評価単位の番号
- `source_phoneme_start`: 元の音素配列における開始位置
- `source_phoneme_end`: 元の音素配列における終了位置（この位置は含まない）
- `expected_units`: 長母音表現を期待母音列へ展開したときの母音数
- `is_long`: 一つの生phoneが長母音表現かどうか
- `phoneme_review_status`: 元のG2P結果に対する読み確認の状態

このファイルには時間境界を含めません。アライナーごとのphone分割数は一致しないことがあるため、区間との照合は`expected_units`を展開した母音列の順序で検証します。期待単位とモデル区間が常に1対1で対応するとはみなしません。
