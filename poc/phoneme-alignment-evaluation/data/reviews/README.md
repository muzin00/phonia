# Assisted reviews

事前に境界候補がマークされた確認画面で、非専門家が選択式で回答した結果をJSONL形式で保存します。

- 対応する音声と発話情報: [`../samples/manifest.jsonl`](../samples/manifest.jsonl)
- 期待音素列: [`../phonemes/expected.jsonl`](../phonemes/expected.jsonl)
- 評価方針: [`../../annotation-guideline.md`](../../annotation-guideline.md)
- UI仕様: [`../../review-ui-spec.md`](../../review-ui-spec.md)

レビュー画面では方式名を隠して候補A、B、Cとして提示し、保存時に実際の方式名との対応を記録します。

## レビューデータの生成

正規化済みアライメントから、UI用データと候補対応表を生成します。

```sh
uv run python scripts/build_review_dataset.py
```

- `review-dataset.json`: UIへ渡す、方式名を含まないデータ
- `candidate-map.json`: 候補IDと方式・モデル情報の対応表

複数方式を比較する場合は、方式ごとの正規化結果を指定します。候補A/B/Cへの割り当ては項目ごとに固定シードから決まり、同じ入力なら再生成しても変わりません。

```sh
uv run python scripts/build_review_dataset.py \
  --candidate mfa=data/alignments/mfa/normalized.jsonl \
  --candidate julius=data/alignments/julius/normalized.jsonl
```

ブラインド評価中は`candidate-map.json`をUIから配信しません。回答の集計時にこの対応表を使って方式名へ戻します。

UIから保存した回答は既定で`review-records.jsonl`へ追記されます。同じレビュー項目を再評価した場合も過去の行は上書きせず、`revision`を増やした新しい行を追加します。UIを再読み込みした場合は、データセットと項目ごとの最新リビジョンが復元されます。
