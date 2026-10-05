# Phase 6: 本人照合の認証性能評価

[Issue #42](https://github.com/muzin00/phonia/issues/42)の評価設計と実装。
2026-10-05に、固定した条件でJVSのvalidation/test各15話者・1,200発話の評価を完了した。
主条件のtest結果は他人受入率0.729%、条件付き本人拒否率2.585%、入力不足を含む本人拒否率4.533%、
統合scoreのEER 1.526%。全条件・件数・信頼区間と限界は[評価結果](evaluation-results.md)を参照する。
ブラウザで確認する場合は[評価結果HTML](evaluation-results.html)を開く。
主条件の指標、入力の内訳、全6図、データ・母音・動作点の条件比較を表示する。
数値と画像はHTML内に埋め込み、ネットワーク接続なしで閲覧できる。

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
protocolの`designed_not_evaluated`は評価前に固定した設計版の状態を残す。
各runの実施状況は`execution.json`に保存し、今回のrunは`completed`である。

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

## 評価を実行する

評価用のPython 3.12環境はこのディレクトリで管理する。numpy・torchは既存APIと同じ版に固定し、
描画用matplotlibと開発用ruffを`uv.lock`に記録する。

```sh
uv sync --project poc/phoneme-verification-evaluation --locked

uv run --project poc/phoneme-verification-evaluation python \
  poc/phoneme-verification-evaluation/scripts/run_evaluation.py all \
  --output-dir artifacts/phoneme-verification-evaluation/<unused-run-id>
```

`all`は`prepare` → `validation` → `freeze` → `test` → `report`の順で実行する。
途中までを確認する場合は`all`を各stage名に置き換え、同じ`--output-dir`で順番に実行する。
`prepare`は既存3フェーズとPhase 6のテスト、ruff、差分チェックを実行し、validationの入力を作る。
`freeze`はvalidationの閾値と実装のchecksumを確認して、testの入力と計画を固定する。
testのencoder推論は`test` stageで初めて実行する。

同じstageを再実行して既存ファイルを上書きすることはできない。
失敗は`failures/`へ記録する。この初期実装は途中stageのresumeを提供せず、
修正後の再実行には新しいrun IDを使用する。test観測後の再実行は独立確認とは扱わない。

## 実装と保存形式

登録・照合の計算はPhase 4/5の公開APIを使用する。
音声、元WAV checksum、frame範囲、モデル・前処理の識別情報をcache keyへ含めた
メモリ内embedding cacheを使い、音声の品質・重複・checksum検査は各APIに残す。
validationでは各話者・roleの本人/別人trialと登録1・5・10区間について、
cacheなしのAPI結果および入力順を逆にした結果との一致を確認する。

`scores/*.jsonl`は全trialのコンパクトな集計用記録。
`phase5-results/*.jsonl.gz`はscoreあり全trialのPhase 5結果JSONとchecksumを保存する。
入力不足は`queries/`と`no_score` trialへ残し、偽のscoreを生成しない。
`evaluation-plan.json`はtest推論前の条件、`execution.json`は生成後の全成果物のhashを記録する。
生成物は上書きせず、モデル・特徴統計・profileの不変も各splitで検査する。

## 実装の検証

```sh
uv run --project poc/phoneme-verification-evaluation python \
  -m unittest discover -s poc/phoneme-verification-evaluation/tests -v
uv run --project poc/phoneme-verification-evaluation ruff check poc/phoneme-verification-evaluation
uv run --project poc/phoneme-verification-evaluation ruff format --check poc/phoneme-verification-evaluation
```

## 保存済み結果からHTMLを作る

```sh
python3 poc/phoneme-verification-evaluation/reports/render_html.py \
  --run artifacts/phoneme-verification-evaluation/jvs-utterance-20261005-v1
```

既定の出力はこのディレクトリの`evaluation-results.html`。
標準ライブラリのみで、保存済み指標と図のchecksumを検査して生成する。推論や閾値の再較正は行わない。
表示用コードは`reports/`へ置き、凍結した評価処理のコードとrun成果物を変更しない。

## 評価の工程

1. 登録区間・発話単位の照合入力・本人/別人trial・入力不足の記録を作成する。
2. Phase 4でprofileを生成し、Phase 5でvalidationのスコアを計算する。
3. 閾値と評価計画を保存し、合成データで集計・閾値適用・リーク検査を確認する。
4. 全条件を固定してtestを実行し、指標・曲線・信頼区間と限界を報告する。

設計書とprotocolを作成しただけでは、最終評価計画を凍結したことにはしない。
Phase 6実装、trial一覧、validation閾値のchecksumがそろった段階で実行計画を固定する。
