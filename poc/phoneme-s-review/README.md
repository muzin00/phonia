# /s/ の学習前レビュー

[学習データ採用手順](../../docs/training-data-adoption.md)に沿って、JVSとCommon Voice日本語の
既存train各70話者から `/s/` の学習候補を自動チェックし、各100件を固定seedで抽出する。
新しい学習はレビューと採用判断の後に行う。

## 対象と固定条件

- 対象はJuliusのラベル `s` のみ。「さ・す・せ・そ」の子音部分で、母音を含む音節全体ではない。
- `sh`、`z`、`ts` は混ぜない。SRC4VCとvalidation/testの音声も使用しない。
- 元WAVは24kHz・mono・PCM16、元アライメントとメタデータ・SHA256を照合する。
- 元音素区間の中だけで最大250msにcenter cropし、実際に使う候補区間の30ms以上・RMS−50dBFS以上を確認する。
- 正規化・無音pad・文脈の追加は行わない。今後の/s/学習はこの固定済み候補manifestを使う。
- 自動チェック後の全候補から各100件を重複なくランダム抽出する。seedは20261010。
- 抽出順は区間IDのseed付きSHA256順、表示順は独立したseed20261011のSHA256順。
- 話者・長さ・隣接音・音量でレビューサンプルを入れ替えない。

設定は[protocol.json](config/protocol.json)。raw・eligible・excluded manifestと元入力のchecksumを
`artifacts/phoneme-s-review/s-200-20261010-v1/` に保存する。
これは前回m/nの音量除外前の予備レビューとは異なり、自動チェック後の正式な採用判断用サンプルである。
各話者の実在人物がコーパス間で重複していないかは確認できていない。

## 起動と確認

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-s-review/prepare_review.py
sh poc/phoneme-s-review/start-review.sh jvs
sh poc/phoneme-s-review/start-review.sh common-voice
```

サーバーはそれぞれ別のターミナルで起動する。

| 対象 | URL | 回答保存先 |
| --- | --- | --- |
| JVS 100件 | http://127.0.0.1:5176/ | `data/jvs-review/review-records.jsonl` |
| Common Voice 100件 | http://127.0.0.1:5177/ | `data/common-voice-review/review-records.jsonl` |

1. 「候補Aだけ再生」で/s/が聞こえるか回答する。
2. 前後100msの文脈と元発話で切り出し位置を確認する。
3. 学習データとして「利用できる」「利用できない」「判断できない」を回答する。
4. 上部の「JSONLへ保存」で回答ファイルへ追記する。

/s/は息が擦れるような音なので、その聞こえ方だけを理由にノイズと判定しない。
元発話と表示された位置を照合して判断する。文脈再生は位置確認の補助であり学習区間には含めない。
回答・現在位置はブラウザへ自動保存されるが、採用判断の記録にはJSONLへの保存も必要である。
JVS・CV・以前のm/nレビューは別のデータセットIDと保存先を使う。

抽出処理は既存の出力先がある場合に停止し、固定したサンプルや人間の回答を上書きしない。
音声はGit対象外のartifactsに置き、区間PCMと元音声の対応をSHA256で保存する。

## 準備済みサンプルと検証

2026-10-10に自動チェック後の抽出とUI起動を完了した。
低音量の除外はRMSによる自動判定であり、人間による無音率やラベル精度の測定ではない。

| コーパス | 自動チェック前 | 候補 | 低音量で除外 | レビュー件数 | レビュー内の話者数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| JVS | 16,608 | 16,499 | 109 | 100 | 54 |
| Common Voice | 10,919 | 10,343 | 576 | 100 | 49 |

JVSの選択区間は40〜180ms（中央値110ms）、CVは30〜250ms（中央値120ms）。
両コーパスとも100件は100個の元発話を含む。人間レビュー・採用判断は未実施。

再現と配信を確認する監査は以下で実行する。`--check-http`には両サーバーの起動が必要。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-s-review/audit_review.py --check-http
```

元入力・生成物31,310ファイルのchecksum、全候補と除外候補の分離、train話者の一致、
200件の固定seed再抽出・元音素境界内center crop・区間PCMとUI時刻の一致を独立に照合した。
全200元音声のHTTP配信checksumとbyte-range応答も確認した。
UIのデータ形式検証、Ruff、シェル構文検査を通過し、ブラウザで両画面の波形と再生ボタンの有効状態を確認した。
監査結果はartifacts内`independent-audit.json`に保存している。監査では人間の回答を作成しない。
