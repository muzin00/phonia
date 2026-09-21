# Expert annotations

必要になった場合に限り、専門家が作成した精密な参照境界をPraat TextGrid形式で保存します。初期評価では作成を必須としません。

- 命名規則: `{utterance_id}.TextGrid`
- 対応する音声と発話情報: [`../samples/manifest.jsonl`](../samples/manifest.jsonl)
- 評価方針: [`../../annotation-guideline.md`](../../annotation-guideline.md)

非専門家による選択式の確認結果はこのディレクトリへ保存せず、`data/reviews/`へ保存します。自動生成された音素境界も方式ごとの派生データとして分離します。
