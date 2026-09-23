# Julius alignment results

Julius Speech Segmentation Toolkitのmonophoneモデルによるモデル固有出力と、共通母音区間へ変換した結果を保存する。

生成方法は[`../../../aligners/julius/README.md`](../../../aligners/julius/README.md)を参照する。

## 成果物

- `input/`: 16 kHz派生WAV、基準音素列、DFA、辞書。Git管理しない
- `raw/*.json`: フレーム番号、時刻、音素ラベル、`n_score`
- `raw/*.log`: Juliusの実行ログ。ローカルパスを含むためGit管理しない
- `normalized.jsonl`: 母音phone区間を結合せず共通表記へ変換した結果
- `run.json`: バージョン、コミット、チェックサム、変換条件、処理時間
- `validation.json`: 期待母音列との一致と時間区間の機械検査結果

Juliusには共通の基準音素列を直接与える。長母音を一つの`o:`へ変換せず、連続する`o o`は二つの原子区間として保持する。
