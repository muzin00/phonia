# Assisted reviews

事前に境界候補がマークされた確認画面で、非専門家が選択式で回答した結果をJSONL形式で保存します。

- 対応する音声と発話情報: [`../samples/manifest.jsonl`](../samples/manifest.jsonl)
- 期待音素列: [`../phonemes/expected.jsonl`](../phonemes/expected.jsonl)
- 評価方針: [`../../annotation-guideline.md`](../../annotation-guideline.md)
- UI仕様: [`../../review-ui-spec.md`](../../review-ui-spec.md)

プロトコルv5のレビュー画面では方式名を隠し、選定対象の一方式が出力した一つの原子母音区間を候補Aとして提示します。保存時に実際の方式名との対応を別ファイルへ記録します。

## レビューデータの生成

正規化済みアライメントから、UI用データと候補対応表を生成します。

```sh
uv run python scripts/build_review_dataset.py
```

- `review-dataset.json`: UIへ渡す、方式名を含まないデータ
- `candidate-map.json`: 候補IDと方式・モデル情報の対応表

個々の区間品質を確認するデータセットは、方式ごとの原子`vowel_interval`から生成します。同じ母音が連続しても結合せず、各区間を単独の候補としてレビューします。

`build_review_dataset.py`はプロトコルv5では一度に一方式だけを受け付けます。複数方式を同じ項目で比較する場合は、期待列上の同じ対象へ対応する元区間IDを明示した派生的な比較データを先に定義します。方式ごとの連番や時刻の近さだけで区間を暗黙に対応付けたり、原子区間を結合した結果を学習データへ戻したりしません。

旧プロトコルの`review-records.jsonl`は履歴として保持します。v5のデータセットはバージョンが異なるため、UIと保存処理は旧回答を復元・集計対象に含めません。

ブラインド評価中は`candidate-map.json`をUIから配信しません。回答の集計時にこの対応表を使って方式名へ戻します。

編集中の回答と現在位置はブラウザの`localStorage`へ自動保存されます。画面上部の「JSONLへ保存」を実行したときだけ、回答済みかつ未出力または変更済みの項目が`review-records.jsonl`へ追記されます。同じレビュー項目を再評価した場合も過去の行は上書きせず、`revision`を増やした新しい行を追加します。UIを再読み込みした場合は、JSONLの最新リビジョンを基準にブラウザ内の編集状態が復元されます。
