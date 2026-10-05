# Phase 6: 本人照合の認証性能評価

[Issue #42](https://github.com/muzin00/phonia/issues/42)の評価設計。
2026-10-05時点で、評価条件と機械可読設定を作成し、validationのメタデータで実行可能性を確認した。
認証性能の測定と評価処理の実装はこれから行う。

## 設計の要点

- 登録は5母音を各10区間とする。各1・5区間は補助比較として扱う。
- 照合は**1発話を1入力**とし、その発話内の利用可能な全母音区間を使用する。
- 同じ母音のcosineを母音内で平均し、5母音を等重みで統合するPhase 5の処理を使う。
- validationで判定閾値を決め、別話者のtestへ固定して適用する。
- 本人・別人の誤判定に加え、5母音不足でスコアを出せない割合も報告する。
- JVSでは別発話での評価を行う。収録日・セッション・端末の変化への性能は評価できない。

具体的な試行・閾値・指標・実装手順は[評価設計](evaluation-design.md)を参照する。
正本の設定は[`config/evaluation-protocol.json`](config/evaluation-protocol.json)。
メタデータ調査の根拠は[実行可能性の確認](validation-feasibility.md)にまとめた。

## 検証用メタデータの調査を再現する

このスクリプトはvalidationのメタデータだけを集計する。
WAVの読込、encoderの推論、閾値の較正、testスコアの計算は行わない。
既存manifestと設定のchecksumを確認し、生成物はGit管理外の`artifacts/`へ置く。

```sh
python3 poc/phoneme-verification-evaluation/scripts/inspect_validation.py \
  --output artifacts/phoneme-verification-evaluation/design/validation-readiness.json
```

保存先は上書きしない。再実行時は未使用の`--output`を指定する。
Python 3.10以降の標準ライブラリだけで動く。認証評価本体はPhase 4・5と同じPython 3.12を使用する。

## 次に実装するもの

1. 登録区間・発話単位の照合入力・本人/別人trial・入力不足の記録を作成する。
2. Phase 4でprofileを生成し、Phase 5でvalidationのスコアを計算する。
3. 閾値と評価計画を保存し、合成データで集計・閾値適用・リーク検査を確認する。
4. 全条件を固定してtestを実行し、指標・曲線・信頼区間と限界を報告する。

設計書とprotocolを作成しただけでは、最終評価計画を凍結したことにはしない。
Phase 6実装、trial一覧、validation閾値のchecksumがそろった段階で実行計画を固定する。
