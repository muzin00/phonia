# Phase 2 stratified vowel review

Issue #17で実施したJulius母音区間の層化レビュー成果物を保存する。

- `sample.jsonl`: 固定seedで抽出した500区間
- `sampling-summary.json`: 抽出集合の条件別件数
- `review-dataset.json`: レビューUIへ渡すデータ
- `candidate-map.json`: 匿名候補AとJuliusの対応
- `review-records.jsonl`: 300件で早期終了した非専門家レビュー
- `review-summary.json`: 最新リビジョンを使った条件別集計

サンプル、UI用データ、集計はプロジェクトのREADMEに記載したコマンドで再生成できる。
人手回答である`review-records.jsonl`は再生成・上書きしない。
