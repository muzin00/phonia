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
比較用profile・閾値・認証性能は未測定。次は3方式×5長さ条件のvalidation pilotを実装・検証する。

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

## 関連資料

- [研究設計のベースライン比較](../../docs/research-design.md#11-ベースライン比較)
- [Phase 6の評価設計](../phoneme-verification-evaluation/evaluation-design.md)
- [Phase 6の評価結果](../phoneme-verification-evaluation/evaluation-results.md)

設計・設定・実行処理・結果はこのディレクトリへ追加し、生成した実験成果物は
Git管理外の`artifacts/phoneme-baseline-comparison/`へ保存する。
