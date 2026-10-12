# 話者・発話拡張セットのレビュー

利用者の「続けて」を受け、選定済みの追加集合から品質確認用UIを準備する。新しいモデルの学習はまだ開始しない。前回のJTubeレビューの回答・採用判断は流用しない。

## 対象と抽出

追加候補11,200発話の4集合それぞれから、固定seedで40話者ラベル×1発話を選び、計160発話を処理する。各集合内で話者をSHA-256順位に並べ、各話者の発話も同じseedの順位で選ぶ。`existing_cv`は追加対象の40ラベル全て、それ以外は各集合のラベルから40を使う。同じ既存JTubeラベルが別の集合に入ることはない。

Common Voiceは配布テキストを使用し、JTubeは既存のWhisper large-v3-turbo固定revisionで日本語を書き起こす。書き起こしは人手の正解ではない。JTubeは30秒超の発話をASR前に除外する。位置合わせは両コーパスとも30秒上限・Julius 10秒上限を適用する。失敗した発話を別の発話へ差し替えない。

前回と同じ24kHzモノラルPCM16、音素変換、Julius、30〜250ms、RMS −50dBFS以上、36音素の抽出条件を使う。各集合の自動チェック後の区間から音素別に巡回割当し、25区間ずつ選ぶ。表示順は固定seedで混ぜる。1集合で25区間を用意できなければ停止し、別集合で補充しない。

| 集合 | UI区間数 | テキスト |
| --- | ---: | --- |
| Common Voice・既存話者の発話追加 | 25 | 配布文 |
| Common Voice・新規話者 | 25 | 配布文 |
| JTube・既存話者の発話追加 | 25 | 自動生成 |
| JTube・新規話者 | 25 | 自動生成 |

11,200発話全体からの単純無作為標本ではない。音素・集合の割当が元の頻度と異なるため、この100区間からコーパス全体の品質率を推定しない。自動チェック成功はラベル正解の証拠ではない。JVSには追加がないため今回のレビューに含めない。

## 起動と確認

```sh
sh poc/phoneme-data-scaling/start-review.sh
```

URL: http://127.0.0.1:5179/

対象タグで4集合を区別できる。区間を聞き、必要に応じて前後100msと元発話で位置を確認する。前後の文脈は学習波形には加えない。任意で回答を「JSONLへ保存」できる。全件の回答・保存を進行の必須条件にはしない。

新しいdatasetIdと保存先`artifacts/phoneme-data-scaling/review100-20261012-v1/review-records.jsonl`を使う。旧レビューの回答は保持する。準備時に回答を生成しない。

## 再現・監査

```sh
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/prepare_review.py select
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/prepare_review.py download
OMP_NUM_THREADS=1 artifacts/phoneme-corpus-scaling/asr-venv/bin/python poc/phoneme-data-scaling/review_asr.py
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/extract_review.py
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/review.py build
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-data-scaling/audit_review.py
node tools/phoneme-review-ui/scripts/validate-example.ts artifacts/phoneme-data-scaling/review100-20261012-v1/review-dataset.json
```

設定は`review-config.json`、保存先は`artifacts/phoneme-data-scaling/review100-20261012-v1`。`freeze.json`で候補集合・抽出用発話・コードを固定し、`asr/freeze.json`でモデルと実行条件を固定する。`runtime-inputs.json`で音素抽出の依存コード・辞書・モデル・実行ファイル・書き起こしを固定する。既存encoderによる特徴・埋め込み計算は行うが、重み更新は行わない。

`pre-qc.json`・`post-qc.json`・`excluded.json`・`extraction-inventory.json`に候補と除外理由を保持する。`selected-segments.json`と`review-dataset.json`がUIへの対応表で、`review-audit.json`に件数・音素内訳・配信波形と区間PCMのhashを保存する。独立監査は元候補からの抽出再現、train限定、各集合25区間、テキスト対応、波形hash、境界、RMSを検査し、`independent-audit.json`へ保存する。

全拡張集合と既存学習・評価データとの波形重複の検査は、このレビュー準備では完了していない。学習前の全量処理で確認する。実在人物のコーパス間重複も不明である。

## 準備結果（2026-10-12）

| 項目 | 件数 |
| --- | ---: |
| 固定発話・話者ラベル | 160 |
| CVの配布テキストを使用 | 80 |
| JTubeの自動書き起こし完了 | 71 |
| JTubeの30秒超過による除外 | 9 |
| 使用可能な音素区間を生成した発話 | 149 |
| 抽出しても使用可能区間がなかった発話 | 2 |
| 自動チェック前の区間 | 7,256 |
| 低RMSによる除外 | 534 |
| 自動チェック後の候補 | 6,722 |
| UIの固定区間 | 100 |
| UIの話者ラベル・元発話 | 74 |
| UIの音素種類 | 29 |

候補プールは32音素を含むが、100件の固定割当では29音素となった。`/dy/`・`/my/`・`/py/`・`/v/` は候補になく、`/f/`・`/u/`・`/z/` は候補にあるが今回の25件ずつの抽出では未選択。全36音素の品質を確認できたとは扱わず、標本は選び直していない。

独立監査とUIスキーマ検証を通過した。HTTPで100件のデータと全74音声の全体取得・Range取得を確認した。Chromeで波形・集合タグ・JTubeの自動生成表示・項目移動・原音声の再生状態の切替を確認し、先頭へ戻してタブを保持した。区間再生の操作も実施したが、これを聴感による品質評価とは扱わない。保存済み回答は0件で、レビュー回答は作成していない。画面はrun配下の`review-ui.png`、HTTP確認は`http-audit.json`に保存した。

準備時、制限環境でMacの音声コーデックへアクセスできず151発話の変換が失敗した。この実行の空manifestとログを`infrastructure-attempts/sandbox-codec-failure/`へ保管し、同じ入力・コード・条件で音声機能へアクセスできる環境から再実行した。データ品質による除外とは区別する。ASRのGPU利用とローカルUIの起動にも実行環境に応じたアクセスが必要。
