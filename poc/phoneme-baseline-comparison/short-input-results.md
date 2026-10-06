# ECAPA短入力の動作確認

これは撤回した[旧protocol 1.0.0](archive/v1/config/comparison-protocol.json)の診断記録。
現行設計は[比較設計2.0.0](comparison-design.md)を参照する。

2026-10-06、比較protocol 1.0.0の利用時間合わせ条件をvalidationで確認した。
**30 msの登録入力は推論エラーとなり、現在の`ecapa_time_exact`条件はそのまま実行できない。**
70 msの登録入力と、診断用の40・50・60・100 ms入力は成功した。
認証score・登録profile・閾値・test推論は作成していない。

## 条件と選び方

[当時の短入力設定](archive/v1/config/short-input-smoke.json)で元の比較protocolとembedding設定のSHA-256を固定した。
同じ公開重み・前処理・CPU float32・batch 1・threads 1を使い、波形のpadding・反復やモデル変更を加えていない。
validationの登録数1・5・10と通常文・異文について、元WAV別の実利用時間が最短・最長の入力を
2つの時間合わせ方式それぞれで選んだ。同点は元WAVパスとcase IDで決め、scoreは使用しない。
全20条件に加え、最短入力の元WAVで30・40・50・60・70・100 msの中央区間を診断用に作った。
この6入力は比較条件へ追加しない。

元音声・モデルのchecksumと音声形式を検証し、24 kHzで中央区間を切り、固定設定で16 kHzへ変換した。
各入力を3回推論し、別プロセスでも同じ確認を繰り返した。

## 実際の比較入力の結果

時間は最短 / 最長。各欄はその時間に対応する選定入力の結果であり、全入力を推論した集計ではない。

| 入力条件 | 境界内母音との時間合わせ | 文脈付き母音との時間合わせ |
| --- | --- | --- |
| 登録1区間／母音 | 0.03 / 0.19秒、最短のみエラー | 0.07 / 0.24秒、両方成功 |
| 登録5区間／母音 | 0.03 / 0.29秒、最短のみエラー | 0.07 / 0.41秒、両方成功 |
| 登録10区間／母音 | 0.03 / 0.40秒、最短のみエラー | 0.07 / 0.52秒、両方成功 |
| 通常文query | 1.33 / 6.85秒、両方成功 | 1.94 / 9.23秒、両方成功 |
| 異文query | 0.67 / 5.59秒、両方成功 | 1.11 / 7.56秒、両方成功 |

20条件のうち17条件が成功、3条件がエラー。成功したembeddingは192次元、有限・非零だった。
エラーは各登録数で選ばれた30 ms入力で、3反復とも同じメッセージとなった。

```text
Padding size should be less than the corresponding input dimension,
but got: padding (4, 4) at dimension 2 of input [1, 128, 4]
```

30 msは16 kHzで480 samples、filterbankの時間方向は4 frames。
エラーのtracebackはECAPAのdilated convolution内のreflection paddingを示している。
これは推論の成立性の制約であり、認証性能の低下を示すscoreではない。

## 長さの診断と再現性

| 入力長 | filterbankの時間frames | 結果 |
| --- | --- | --- |
| 30 ms | 4 | 同じpaddingエラー |
| 40 ms | 5 | 成功 |
| 50 ms | 6 | 成功 |
| 60 ms | 7 | 成功 |
| 70 ms | 8 | 成功 |
| 100 ms | 11 | 成功 |

診断の40 ms成功は1つの元音声についての結果であり、モデルの一般的な最低入力長は決定していない。
成功した全22入力のembeddingは3反復と別プロセス間でbitwise一致した。
選定入力一覧と成功embedding配列のSHA-256もプロセス間で一致し、エラーメッセージも一致した。
モデルのparameters・buffers・公開ファイルは確認前後で不変だった。
環境はPython 3.12.13、torch/torchaudio 2.11.0、SpeechBrain 1.1.1、NumPy 2.5.3、macOS arm64。

## 登録条件への影響

固定済みvalidation登録metadataから、元WAV別の境界内時間予算が30 msとなる件数を数えた。
この表は全元WAVの推論結果ではなく、失敗した入力長が存在する登録条件の集計である。

| 各母音の登録数 | 30 msを含む話者profile数 | 30 msの元WAV数 / 条件内の元WAV延べ数 |
| --- | --- | --- |
| 1 | 9 / 15 | 13 / 69 |
| 5 | 14 / 15 | 45 / 311 |
| 10 | 14 / 15 | 50 / 497 |

登録元WAVごとのembeddingを全て等重みで使う現行規則では、これらの短入力を含む条件に対応できない。
短い元WAVの除外、波形の延長、反復、失敗の`no_score`への変換は行わなかった。
この2つの時間合わせ補助条件は撤回し、同じ元発話と秒単位の照合波形制限へ設計を改訂した。
主比較3方式の入力規則はこのエラーの影響を受けないが、比較用profile・scoreの実装検証は未完了である。
旧protocolは初回設計の記録として保持し、test推論・実行計画の凍結へは進んでいない。

## 再現と成果物

```sh
HF_HUB_OFFLINE=1 uv run --project poc/phoneme-baseline-comparison --locked python \
  poc/phoneme-baseline-comparison/scripts/run_short_input_smoke.py \
  --model-dir artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0 \
  --output-dir artifacts/phoneme-baseline-comparison/<unused-run-id>
```

診断scriptは推論のRuntimeErrorを各入力の記録へ残して、他の診断入力も確認する。
認証評価での停止規則は変更しない。終了コード0は診断記録の完了を意味し、
比較処理へ進めるかはreportの`summary.comparison_ready`で判定する。本runの値は`false`。
音声形式やchecksum、embeddingの異常、反復の不一致は診断自体を停止する。

保存先は`artifacts/phoneme-baseline-comparison/ecapa-short-input-20261006-v1/`と
`ecapa-short-input-20261006-v1-recheck/`。選定元音声・区間・anchor、成功embedding、失敗traceback、
モデル・実装・環境・設定のchecksumを保存した。
初回runには`cross-process-check.json`と`metadata-impact.json`も保存している。
各runの`implementation-snapshot/`に当時の実装・設定を保存した。
現行の短入力診断設定は保存済み旧protocolを参照し、同じ切り出し条件を再現する。

| ファイル | SHA-256 |
| --- | --- |
| 初回`short-input-report.json` | `fea7987e2ede599c5ae29f10f0d67d2d42c1e1267ce01d2f921a60a36ba4ef73` |
| 再確認`short-input-report.json` | `449fd189706489fe00f8469d04de3f346b74e7f8e33a069044ed9e9856ebbe0e` |
| 共通`selected-inputs.json` | `6b07103a818271e8de68b262a74bc686313f50258d77019e9c7116f0392eba84` |
| 共通`embeddings.npy` | `afaa6ccdf95313891fa456d520b6af386aadbf45d094c97f16d16a56516ae081` |
| 初回`metadata-impact.json` | `3f2009c39407d340d2846126e198aef0d75297617e4e12988a3e0ee2a410a6a6` |

選定の順序独立性、時間予算の一致、test入力の拒否、診断入力の区別を含む20テストとruff検査を通過した。
