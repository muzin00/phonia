# Wav2Vec2 assisted review

Wav2Vec2 phoneme CTC forced alignmentが出力した原子母音区間138件を、既存の
レビューUIで確認するためのデータを保存する。

- `review-dataset.json`: 方式名を伏せてレビューUIへ渡すデータ
- `candidate-map.json`: 候補AとWav2Vec2の対応表

レビューデータは次のコマンドで再生成できる。

```sh
uv run python scripts/build_review_dataset.py \
  --candidate wav2vec2=data/alignments/wav2vec2/normalized.jsonl \
  --dataset-id jvs-vowel-alignment-review-wav2vec2 \
  --dataset-version 1 \
  --output data/reviews/wav2vec2/review-dataset.json \
  --mapping-output data/reviews/wav2vec2/candidate-map.json
```

今回は少数区間の参考確認だけを行い、全138件の定量評価は実施していない。参考確認
では明らかな品質不足は感じられず、聴感上はMFAと同程度という印象だった。正式な
レビュー記録として扱う場合は、UIの「JSONLへ保存」で`review-records.jsonl`を
生成し、確認件数と選定方法を併記する。
