# JTubeSpeech-ASV 学習前レビュー

利用者からの「レビューUIの準備」「100個くらいランダムで」の依頼に対応する。追加学習は行わず、利用者による品質確認・採用判断を待つ。

## 対象とランダム抽出

- 元集合: 固定済みJTubeSpeech-ASV train 140話者ラベル × 50発話 = 7,000発話。validation/testは使わない。
- 第1段階: 各話者の50発話からseed `20261112`のSHA-256順位で1件ずつ選ぶ。140発話をローカルASRにかける。
- 処理上限: 最初の上限なし処理が50.508秒の発話の位置合わせ中に終了コード137で停止したため、Juliusへの入力発話を30秒以下、実行時間を10秒以内に制限した。上限超過・タイムアウトは固定140発話内の失敗として残し、代替発話は選ばない。長い発話の品質は今回のUIでは確認できない。
- 第2段階: 同じ音素変換・Juliusアライメント・24kHz PCM16処理を適用。30–250ms、RMS −50dBFS以上、現行36音素の条件を満たす候補から100区間を選ぶ。
- 音素内はseed `20261113`のSHA-256順位でランダム抽出。音素順も同seedで固定し、1件ずつ巡回割当する。候補不足の音素はその件数までとし、残りを他音素に割り当てる。
- 表示順はseed `20261114`でシャッフル。人手の感想や採点結果を見て差し替えない。

**7,000発話の全区間からの単純無作為抽出ではない。** 140発話という第1段階の固定範囲から音素別に抽出する。レビュー全体の割合はこの標本の記述値であり、音素頻度が異なるコーパス全体の品質率とは区別する。話者ラベルが実在人物に正しく対応することや、音声内の話者の一貫性も自動抽出の成功だけでは保証しない。

テキストはWhisperによる機械生成で、人手の正解テキストではない。UIのタイトル・対象タグ・設問に明記する。長い音素は中心250msまでに制限し、UIの区間再生は抽出された波形と同じフレーム範囲を使う。前後100msは位置確認用で学習波形には加えない。Runtime由来の`split: pilot`を保持し、元データのtrain分割は別の`source_split`で追跡する。

## 起動とレビュー

```sh
sh poc/phoneme-corpus-scaling/start-review.sh
```

URL: http://127.0.0.1:5178/

1. 区間を聞き、表示された音素との対応を回答する。
2. 必要に応じて前後の文脈付き再生・元発話で確かめる。
3. 学習用として利用できるか回答する。
4. 上部の「JSONLへ保存」で回答を保存する。途中保存・再開も可能。

他のレビューと異なるdatasetIdと回答保存先を使う。保存先は`artifacts/phoneme-corpus-scaling/review-100-20261011-v1/review-records.jsonl`。回答は人間がUIで行い、準備処理で自動回答しない。

## 再現・監査

以下は準備済みrunの初回実行経路。新たな抽出条件ではrun_directoryを変え、回答済みのサンプルを書き換えない。

```sh
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/review.py select
OMP_NUM_THREADS=1 artifacts/phoneme-corpus-scaling/asr-venv/bin/python poc/phoneme-corpus-scaling/review_asr.py
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/review_extract.py
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/review.py build
```

`review-config.json`で範囲・seedを固定。run配下に次を保存する。

| ファイル | 内容 |
| --- | --- |
| `freeze.json`, `selected-utterances.json` | 元manifest・選択発話・コードのhash、seed |
| `asr/freeze.json`, `asr.json` | モデル・パッケージ・デコード設定・ASR結果・失敗 |
| `alignment-resource-policy.json`, `extraction-freeze.json`, `extraction-inventory.json` | 抽出条件と140発話の処理結果 |
| `pre-qc.json`, `post-qc.json`, `excluded.json` | 自動チェック前後の区間と除外理由 |
| `selected-segments.json`, `review-dataset.json` | 固定100区間とUI入力の対応 |
| `review-audit.json` | 件数、話者数、音素別内訳、元WAVと切り出しPCMのhash |
| `media/` | 選択区間の元発話WAVのみを配信 |

今回の初回準備中、UIの期待単位数を「発話全体の音素数」から「対象の原子音素1件」へ修正した。ASR開始時点のfreezeとコードは`selection-asr-freeze.json`・`review-preparation-original.py`・`review-asr-original.py`として保存し、現行freezeに修正理由を記録した。発話・区間の選び方、ASR条件に変更はない。`asr/freeze.json`のselection hashは保存済みの旧freezeを指す。

監査は `poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-corpus-scaling/audit_review.py` で再実行できる。

音素の品質・学習の改善傾向はまだ判定していない。結果はレビュー後に集計する。

## 今回の準備結果

| 項目 | 件数 |
| --- | ---: |
| 固定したtrain発話・話者ラベル | 140 |
| 自動書き起こし完了 | 140 |
| 音素候補を生成できた発話 | 126 |
| 候補を生成できなかった発話 | 14 |
| 自動チェック前の音素区間 | 7,298 |
| 低音量で除外 | 496 |
| 自動チェック後の候補 | 6,802 |
| UIの固定サンプル | 100 |
| UIに含まれる話者ラベル・発話 | 60 |
| UIに含まれる音素 | 33 |

14発話の内訳は、30秒超過6件、10秒タイムアウト1件、アライメント不成立7件。発話の差し替えはない。自動チェックの通過を品質の合格とは扱わない。

現行36音素のうち `/by/`・`/dy/`・`/v/` は候補がなく未確認。希少な `/my/`・`/py/` は各1件、`/gy/` は2件、それ以外は各3〜4件をレビューする。この100件だけで全36音素の採用が確認できたとはしない。

標本再現・train限定・区間PCMのhash・フレーム境界・RMSの独立監査と、UI入力スキーマ検証を通過した。HTTPでは100件のデータと全60音声の全体取得・Range取得、回答が未保存であることを確認した。ブラウザ自動操作機能が起動できず、画面の目視と実際の聴取は未実施。回答は0件で、品質判定・追加学習には進んでいない。
