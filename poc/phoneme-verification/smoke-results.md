# Phase 5 動作確認結果

2026-10-05に、Phase 4の保存済み`jvs007`プロファイルとPhase 3の固定配布bundleを使用し、
登録用とは別のvalidation音声からスコアを計算した。
これは実装の動作検証であり、認証精度の評価ではない。

## 条件

- encoder: `statistics_mlp`、seed `20260926`、70話者学習の固定checkpoint
- 登録: `jvs007`、各母音10区間、Phase 4で保存済みのプロファイル
- 照合: 全5母音を各5区間、合計25区間。元WAV round-robinとseed `20261005`で選択
- 音声は複数発話にまたがる。各試行の元WAV数は本人19、別人23、cross-text本人15
- 計算: 区間cosineの母音内平均、5母音の等重み平均
- 実行: Python 3.12.13、NumPy 2.5.3、PyTorch 2.14.1、macOS arm64、CPU / float32 / batch=1
- train話者・最終testは使用せず、閾値や統合重みも調整していない

## スコア

| 照合ケース | 照合話者 / role | a | i | u | e | o | 統合 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 本人 | jvs007 / verification | 0.533478 | 0.682655 | 0.561766 | 0.650493 | 0.587908 | 0.603260 |
| 別人 | jvs016 / verification | 0.471902 | 0.297633 | 0.213877 | 0.109209 | 0.207723 | 0.260069 |
| 発話内容が異なる本人 | jvs007 / cross_text_verification | 0.585552 | 0.471622 | 0.374918 | 0.629226 | 0.685642 | 0.549392 |

除外区間は全ケースで0。登録と照合のWAV checksumは全ケースで重複していない。
この3例では本人スコアが別人より高かったが、受入れ・拒否の判断は行っていない。
母音の区間数や複数発話のまとめ方が異なる条件へ、そのまま性能を一般化しない。

## 再現性と不変性

- 全3ケースを入力順を逆にして再実行し、保存した結果ファイルが完全一致した。
- 各結果の再読込でchecksum、件数、区間・母音・統合スコアの整合性を検査した。
- checkpoint、run情報、特徴統計ファイルのSHA-256は実行前後で一致した。
- encoderの全state tensor、特徴統計、登録済みプロファイルファイルも変わっていない。
- encoderはevalモードで勾配無効。parameter更新、再学習、プロファイル更新を行っていない。

Phase 5の合成データ・CLIテスト16件、Phase 4の12件、Phase 3の35件、計63件が通過した。
ruff check、ruff format --check、git diff --checkも通過した。

## 保存した成果物

Git管理外の`artifacts/phoneme-verification/smoke/`に以下を保存した。

- `genuine-input.json` / `genuine-result.json` / `genuine-result-repeat.json`
- `impostor-input.json` / `impostor-result.json` / `impostor-result-repeat.json`
- `cross-text-genuine-input.json` / `cross-text-genuine-result.json` / `cross-text-genuine-result-repeat.json`
- `verification.json`: 上記のスコア、入力の由来、再現性・不変性の検査結果

使用したプロファイルの内容checksum:
`50642e64d0165b1ebd8c2ee20efb6a227ac854dada80bfe5e83c56054d742ffe`

再実行コマンドは[README](README.md#実データの動作確認を再実行する)を参照する。
後日のレビューでは[初期設計](design.md)を確認し、Phase 6の試行定義・閾値決定用データ・
評価用データを決めて認証性能の検証へ進む。
