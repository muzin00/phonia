# Assisted reviews

事前に境界候補がマークされた確認画面で、非専門家が選択式で回答した結果をJSONL形式で保存します。

- 対応する音声と発話情報: [`../samples/manifest.jsonl`](../samples/manifest.jsonl)
- 期待音素列: [`../phonemes/expected.jsonl`](../phonemes/expected.jsonl)
- 評価方針: [`../../annotation-guideline.md`](../../annotation-guideline.md)
- UI仕様: [`../../review-ui-spec.md`](../../review-ui-spec.md)

レビュー画面では方式名を隠して候補A、B、Cとして提示し、保存時に実際の方式名との対応を記録します。

UIから保存した回答は既定で`review-records.jsonl`へ追記されます。同じレビュー項目を再評価した場合も過去の行は上書きせず、`revision`を増やした新しい行を追加します。UIを再読み込みした場合は、データセットと項目ごとの最新リビジョンが復元されます。
