# Julius assisted review

Juliusが出力した原子母音区間138件に対する、プロトコルv2の支援レビュー成果物を保存する。

- `review-dataset.json`: 方式名を伏せてレビューUIへ渡すデータ
- `candidate-map.json`: 候補AとJuliusの対応表
- `review-records.jsonl`: 非専門家による全138件のレビュー結果

レビューデータは次のコマンドで再生成できる。

```sh
uv run python scripts/build_review_dataset.py \
  --candidate julius=data/alignments/julius/normalized.jsonl \
  --dataset-id jvs-vowel-alignment-review-julius \
  --dataset-version 1 \
  --output data/reviews/julius/review-dataset.json \
  --mapping-output data/reviews/julius/candidate-map.json
```

`review-records.jsonl`は再生成せず、人間がレビューUIで保存した履歴を保持する。
