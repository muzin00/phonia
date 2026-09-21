# Assisted reviews

事前に境界候補がマークされた確認画面で、非専門家が選択式で回答した結果をJSONL形式で保存します。

- 対応する音声と発話情報: [`../samples/manifest.jsonl`](../samples/manifest.jsonl)
- 期待音素列: [`../phonemes/expected.jsonl`](../phonemes/expected.jsonl)
- 評価方針: [`../../annotation-guideline.md`](../../annotation-guideline.md)
- UI仕様: [`../../review-ui-spec.md`](../../review-ui-spec.md)

レビュー画面では方式名を隠して候補A、B、Cとして提示し、保存時に実際の方式名との対応を記録します。
