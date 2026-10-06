# Phase 7: 既存話者認証方式とのベースライン比較

[Issue #45](https://github.com/muzin00/phonia/issues/45)に沿って、Phase 6の母音方式と
発話全体を入力する既存speaker embedding方式、前後の文脈を含む母音方式を比較する。

発話全体方式にはSpeechBrainの`speechbrain/spkrec-ecapa-voxceleb`を選び、
2026-10-06にvalidationの5話者・11発話でembedding抽出の動作確認を完了した。
候補、選定理由、学習条件の差、残る設計事項は[モデル選定](model-selection.md)を参照する。
192次元の有限・非零embedding、同一入力の反復と別プロセス間の完全一致、モデルの不変を確認した。
CPU推論は平均約44 ms/発話。[動作確認結果](smoke-results.md)に条件・計測範囲を記録した。
同日に[比較設計](comparison-design.md)と[比較protocol 2.0.0](config/comparison-protocol.json)へ改訂した。
境界内母音・前後各20 ms付き母音・発話全体ECAPAへ同じ元発話を渡す主比較と、
照合波形を最大1・2・3・5秒へ制限する補助比較を定義した。登録profileは全長さ条件で固定する。
[改訂設計の入力確認](query-window-readiness.md)で、共通区間と母音の充足・元発話不足を確認した。
[validation pilot](validation-pilot-results.md)で7話者・12照合発話、3方式×5長さ条件×登録数1・5・10を実行した。
63個のprofileと3,780件の照合レコードを生成し、既存APIとの一致・母音不足の扱い・再現性を確認した。
[全validation](validation-results.md)では15話者・1,200queryの810,000件を実行し、
135個のprofileとnative・共通入力の計270閾値を保存した。
実行条件を凍結し、固定閾値でtestの全810,000件を評価した。
全360条件・10,000回の話者CI・792 paired差・60図のPNG/PDFまで完了した。
主条件の全入力FRRは母音2方式4.533%、ECAPA 0.000%で、ECAPAが優位だった。
[比較結果・信頼区間・限界・Issue完了条件](evaluation-results.md)と[全条件HTML](evaluation-results.html)を参照する。

旧方式の[短入力確認](short-input-results.md)では30 ms登録入力がpaddingエラーとなった。
70 msの推論成功も認証精度の根拠にはならず、元WAVごとの利用時間合わせ2条件を撤回した。
初回設計・設定は[archive/v1](archive/v1/README.md)、主比較の元入力量は
[元発話入力確認](validation-readiness.md)へ保持する。

## 旧設計のECAPA短入力診断を再現する

```sh
HF_HUB_OFFLINE=1 uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/run_short_input_smoke.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-run-id>
```

保存済みprotocol 1.0.0のvalidation時間合わせ入力から条件別の最短・最長を選び、固定長の診断入力も確認する。
各入力の推論エラーを保存し、成功embeddingの反復一致とモデルの不変を検証する。
この診断は認証評価とは別であり、失敗をscoreへ変換しない。
結果・再現性・登録条件への影響は[短入力確認結果](short-input-results.md)を参照する。

## 改訂設計の入力確認を再現する

```sh
python3 poc/phoneme-baseline-comparison/scripts/inspect_query_windows.py \
  --output-dir artifacts/phoneme-baseline-comparison/design/<unused-run-id>
```

標準ライブラリだけで、固定済みvalidation metadataのchecksumと、全長さ条件の共通query集合、
区間の入れ子・境界外母音の除外・文脈clip・実利用時間を確認する。音声・認証scoreは読み込まない。
詳細な集計は[改訂設計の入力確認](query-window-readiness.md)を参照する。

## validation pilotを再現する

母音推論にはPhase 6環境、ECAPA推論にはこのプロジェクトの環境を使う。
両プロジェクトで`uv sync --locked --python 3.12`を済ませ、取得済みのECAPAモデルを指定する。

```sh
HF_HUB_OFFLINE=1 uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/run_validation_pilot.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-pilot-run-id>
```

[pilot設定](config/validation-pilot.json)に従い、scoreを見る前にmetadataだけで発話を選ぶ。
登録profileは全長さ条件で固定し、validationの選択話者間で本人・他人照合を実行する。
境界内母音は全長さ条件でPhase 5公開APIと照合し、fullではPhase 6保存結果とも比較する。
文脈付き母音は元coreの品質検証を通してから拡張し、拡張波形に品質フィルタを追加しない。

各runへ`inputs.json`、`preflight.json`、方式別のembedding・profile・score・不変検査と
統合`pilot-report.json`を保存する。実際の入力frame、PCM・元音声・モデル・実装のchecksumを保持する。
失敗時は`failure.json`を残し、同じ保存先への上書きを拒否する。
別プロセスで同じ処理を再実行し、`inputs.json`、`scores.jsonl`と各workerの
`embeddings.npy`・`embedding-inputs.jsonl`・`profiles.jsonl`・`scores.jsonl`のSHA-256を比較する。
経過時間を含むreportの一致は要求しない。pilotでは閾値を校正せず、test音声・scoreを読み込まない。

再実行後、Phase 6環境で次の検査を行う。両runの数値成果物を比較し、cacheを使わない
既存登録・照合APIで全境界内profile・score・`no_score`を再計算する。

```sh
uv run --project poc/phoneme-verification-evaluation --locked python \
  poc/phoneme-baseline-comparison/scripts/check_validation_pilot.py \
  --reference-run artifacts/phoneme-baseline-comparison/<first-pilot-run-id> \
  --recheck-run artifacts/phoneme-baseline-comparison/<second-pilot-run-id> \
  --output artifacts/phoneme-baseline-comparison/<second-pilot-run-id>/validation-pilot-checks.json
```

## 全validationのscore生成と閾値校正

pilotの後、[validation設定](config/validation.json)で全15話者・1,200queryを実行する。
3方式×5長さ条件×登録数1・5・10の810,000件を圧縮JSONLへ逐次保存する。

```sh
HF_HUB_OFFLINE=1 uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/run_validation.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-validation-run-id>
```

母音・ECAPAはpilotと同じ専用環境で実行し、全固有embeddingを3回反復する。
入力frameを含むcache識別はpilotと同じまま、音声cacheを16元WAVまでに制限する。
境界内母音の全45profileとfullの全54,000trialをPhase 6と比較し、
各query・長さ条件の登録10で本人1件・他人1件を公開APIと照合する。
数値・欠測・ID・ラベル・profile固定・query境界・元音声checksumの不一致は停止する。

方式別に`profiles.jsonl`、`scores.jsonl.gz`、`embeddings.npy`と入力追跡情報を保存する。
全件のID・ラベル・coverageを検査後、通常発話だけからnativeと共通入力の閾値を別々に決める。
各方式・登録数・長さ条件についてFAR 1%・0.1%・EER動作点を校正し、
別テキスト発話へ同じ閾値を固定適用する。母音方式のfull/native閾値はPhase 6とも比較する。

`validation-thresholds.json`へ閾値、`validation-metrics.json`へcoverage・分母・検証用指標、
`validation-curves.jsonl.gz`へ条件別ROC/DETの点、`validation-report.json`へ実行検査を保存する。
いずれかの話者・roleにscoreがない条件は条件付き指標とEER/ROC/DETを評価不能とし、
全入力FRR・FARと件数を残す。元発話5秒以上の固定集合にもnativeの閾値を適用し、再校正しない。
このvalidation stageではtest音声・scoreを読み込まない。後続の凍結・test・bootstrapも完了している。

## 凍結からtest・全条件レポートまで再現する

全validationが完了し、[test設定](config/test.json)が参照する成果物のSHA-256が一致する状態で実行する。
母音環境にはレポート用のmatplotlibが含まれるため、最上位runnerもPhase 6環境で起動する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-baseline-comparison/matplotlib-cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
uv run --project poc/phoneme-verification-evaluation --locked python \
  poc/phoneme-baseline-comparison/scripts/run_test.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-test-run-id>
```

runnerはunit test・lint・formatを通し、testの登録/query/trial metadata、元WAVのchecksumとheader、
共通query区間・anchor・実利用量、実装のsnapshot、依存lock、モデル・特徴統計、全validation成果物、
既存test結果の期待checksum、resource planを`execution-freeze.json`へ固定する。
10,000回のPCG64話者抽出も推論前に保存する。test scoreの内容は凍結後にだけ読み、既存方式の全件一致検査に使う。
元音声のheader/checksum検査は推論前の入力検査であり、testによる閾値調整は行わない。
validation専用runnerのtest拒否は保持し、testには専用の凍結確認と音声readerを使う。

母音とECAPAをそれぞれのlock環境で実行し、固有embeddingごとに3回の完全一致を確認する。
全45境界内profile・fullの全54,000trialをPhase 6と照合し、各長さ条件の公開API一致も検査する。
閾値の再校正は行わず、validationの通常発話で決めた270閾値をtestの通常・別テキストへ適用する。
`test-metrics.json`には全180 test条件、条件付き/全入力指標・coverage、10,000回のCIとpaired差を保存する。
話者がscoreを取得できない場合も保持し、条件付きの精度・曲線・CIを評価不能とする。
bootstrapの分母が0になるreplicateは引き直さず、NPZのNaNと区間の有効/未定義件数に残す。

`report/`にはvalidationとtestの全360条件、3動作点のCSV、元発話長と実利用時間の層別CSV、
paired差CSV、全条件のROC/DET・長さ曲線・元発話5秒以上の固定集合のPNG/PDFとHTMLを保存する。
元発話長の層と固定集合ではnative閾値を流用し、層内で校正しない。
短縮条件のcommon集合はcapごとに変化するため、そのcap−full差には集合変更も含むと明記する。

最後にraw scoreから全試行件数・固定閾値・欠測分母を独立に再集計し、全CIのpercentile、
評価不能条件と出力hashを検査して`test-checks.json`へ保存する。
`test-report.json`の`status=completed`が全工程の成功を示す。失敗runは保存し、上書きしない。
artifactはGit管理外にあり、再実行にはJVSデータ、Phase 3/6の固定成果物と指定validation runが必要になる。

主要ROC/DETの全点版と読みやすい軸ラベルを再生成する場合は、test完了後に次を実行する。
数値成果物を変更せず、別の未使用ディレクトリへPNG/PDFとsource hashを保存する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-baseline-comparison/matplotlib-cache" \
uv run --project poc/phoneme-verification-evaluation --locked python \
  poc/phoneme-baseline-comparison/scripts/render_publication_roc.py \
  --run-dir artifacts/phoneme-baseline-comparison/<completed-test-run-id> \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-publication-run-id>
```

## 表形式のHTMLを再生成する

[evaluation-results.html](evaluation-results.html)をブラウザで開くと、主要結果、比較条件の差、方式差、長さ別の性能を表で確認できる。
全360条件・3動作点は、データ・発話条件・入力集合・登録数・照合長・動作点・方式で絞り込み、並べ替えられる。
件数・CI・閾値の詳細表示、抽出結果のCSV出力、全paired差と時間層別の表も含む。
表のデータはHTML内へ埋め込んでいるため、ローカルファイルとして開ける。図とCSVのリンクには同じディレクトリのファイルを使う。

```sh
python3 poc/phoneme-baseline-comparison/scripts/build_results_html.py
node poc/phoneme-baseline-comparison/scripts/check_results_html.mjs
```

builderは保存済みの3 CSVから生成する。推論・閾値校正・bootstrapの再実行は不要。
検査scriptはブラウザを起動せず、元CSVのhash、0とNEの区別、主要件数、発話条件切り替え、フィルター、
並べ替え、ページ送り、共通集合のcoverage、層別、詳細表示、CSV出力の処理を確認する。

## ECAPAの動作確認を再現する

専用のPython 3.12環境を使用する。torchとtorchaudioは2.11.0に揃え、
SpeechBrain 1.1.1、huggingface-hub 2.1.1を含む依存関係を`uv.lock`へ固定した。
Phase 6の環境とは別に管理する。

```sh
uv sync --project poc/phoneme-baseline-comparison --locked --python 3.12

uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/download_ecapa.py \
  --output-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0

HF_HUB_OFFLINE=1 uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/run_embedding_smoke.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-run-id>
```

モデルの取得は版を固定した公開ファイルのダウンロードだけを行う。
取得したファイルを[設定](config/embedding-smoke.json)のSHA-256と照合し、JVS音声を送信しない。
モデルディレクトリが存在する場合は再取得せず、動作確認へ進む。
モデルディレクトリと各runの出力は上書きしない。失敗後の再実行にも未使用の保存先を指定する。

推論処理はvalidationの元WAVだけを読み、PCM16を`32768.0`で割ったfloat32波形を
torchaudioの固定sinc設定で24→16 kHzへ変換する。VAD、crop、RMS正規化は追加しない。
辞書順の先頭3話者について各roleの先頭発話を選び、各query roleの最短発話を追加する。
各発話を3回推論し、embedding・元音声・モデル・実装・環境の記録を保存する。
cosine行列は数値確認用であり、認証性能の評価結果ではない。

別プロセスの再現性を確認する場合は別のrun IDでもう一度実行し、
`selected-inputs.json`と`embeddings.npy`のSHA-256が一致することを確認する。

## 実装の検証

```sh
uv run --project poc/phoneme-baseline-comparison --locked python \
  -m unittest discover -s poc/phoneme-baseline-comparison/tests -v
uv run --project poc/phoneme-baseline-comparison --locked ruff check poc/phoneme-baseline-comparison
uv run --project poc/phoneme-baseline-comparison --locked ruff format --check poc/phoneme-baseline-comparison
```

テストはtest話者・WAVへの誤った参照、splitの重複、欠落role、元WAVの改変・形式不一致、
root外参照、モデルとmanifestを同時に改変した場合の検出を確認する。
比較用の区間clip・center crop、文脈の重複計数、元WAV別の利用時間一致も確認する。
短入力の選定順序独立性、時間予算の一致、test入力拒否、診断入力の区別も確認する。
現行の共通照合波形の入れ子、短い元発話の無延長、境界をまたぐ母音の除外、
文脈の共通境界clip、full条件の既存geometry一致も確認する。
pilotのテストは母音ごとの等重み、query平均の非正規化、母音欠損、非有限・零vectorの拒否、
実入力区間を使うcache、反復不一致の拒否、同じ波形でも長さ条件ごとに異なるscore IDを確認する。
全validationのテストは通常発話以外での閾値校正の拒否、同点のFAR制御、欠測を含む分母、
scoreがない話者の評価不能表示、別テキストでの閾値固定、圧縮保存の再現性と上書き拒否を確認する。
凍結/testのテストは、freeze欠落・固定ファイル改変・split違反の拒否、validationとのscore計算の一致、
同点を含むbootstrap EERと複製trialの独立計算の一致、genuine/impostorの重み、欠測と条件付きCIの評価不能、
未定義replicateの件数、全長さ条件のpaired抽出、duration区間境界と空集合の保持を確認する。

## 関連資料

- [研究設計のベースライン比較](../../docs/research-design.md#11-ベースライン比較)
- [Phase 6の評価設計](../phoneme-verification-evaluation/evaluation-design.md)
- [Phase 6の評価結果](../phoneme-verification-evaluation/evaluation-results.md)

設計・設定・実行処理・結果はこのディレクトリへ追加し、生成した実験成果物は
Git管理外の`artifacts/phoneme-baseline-comparison/`へ保存する。
