# Phase 6設計の実行可能性: validationメタデータ

2026-10-05に、既存の原発話manifestとPhase 3入力manifestからvalidationだけを集計した。
音声推論やscore計算は行っていない。testの母音カバレッジを見て設計を決めてはいない。

## 発話内の母音カバレッジ

| role | 原発話数 | 利用可能区間がある元WAV数 | 全5母音・各1区間以上 | 各5区間以上 | 各10区間以上 |
| --- | ---: | ---: | ---: | ---: | ---: |
| enrollment | 750 | 749 | 728 | 248 | 0 |
| verification | 750 | 750 | 735 | 289 | 13 |
| cross_text_verification | 450 | 450 | 392 | 34 | 2 |

主要照合は全15話者でそれぞれ49/50発話が全5母音を持つ。カバレッジは735/750 = 98%。
cross-textは1話者22〜28/30発話、合計392/450 = 約87.11%。
登録側の1発話に全5母音がある必要はない。各母音を別の登録発話から集めるPhase 4の手順を維持する。
`jvs058`の登録roleには元発話50件中、利用可能区間のあるWAVが49件あり、
元発話manifestを分母にする必要があることも確認した。

## 初期設計への反映

主照合条件は1発話内の全利用可能区間を使用し、母音ごとの固定5・10区間を要求しない。
各5区間必須にすると主要照合は289/750発話しか対象にできず、同じ1発話条件でも
長い発話へ偏るため、今回の主条件にしない。
5母音不足は明示的なno_scoreとし、条件付きの認証指標と全入力の拒否率を分けて報告する。

このメタデータに基づくvalidationの予定trial数は、登録数ごとに次となる。

| role | scoreありgenuine | scoreありimpostor | no_score genuine | no_score impostor |
| --- | ---: | ---: | ---: | ---: |
| verification | 735 | 10,290 | 15 | 210 |
| cross_text_verification | 392 | 5,488 | 58 | 812 |

人数15、1queryあたり1 genuine + 14 impostorとして算出した。
実行時のPCM品質検査でさらに利用不能区間が見つかった場合は、実測coverageを正本とする。
validationの主要照合impostor 10,290件なら、FARの1件分は約0.009718%。
FAR 0.1%の較正でも許容誤受入れは10件程度であり、低FARの保証とは解釈しない。

## 収録条件の制約

validationの原発話1,950件は、すべて`session_id: null`。
JVSの収録日・個別セッション・マイクIDが確認できないという既存調査と一致する。
異文発話を利用できることと、別セッションへ一般化できることは区別する。
今回のJVS評価を開始するために新しい録音は不要だが、外部条件の評価は別のデータが必要である。

## 再現

[READMEのコマンド](README.md#検証用メタデータの調査を再現する)を使う。
機械可読結果は`artifacts/phoneme-verification-evaluation/design/validation-readiness.json`。
元manifestと規則のchecksumを[`config/evaluation-protocol.json`](config/evaluation-protocol.json)へ記録した。
