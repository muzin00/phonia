# 学習前の /m/・/n/ 音素区間レビュー

今後の採用判断は[学習データの採用手順](../../docs/training-data-adoption.md)に従う。
このディレクトリのサンプルは音量による除外前の予備レビューであり、正式採用前には自動チェック後の学習候補も確認する。

追加予定の鼻音区間が聞こえるか、境界がずれていないかを既存のローカルレビューUIで確認する。
JVS・SRC4VCの既存train各70話者のアライメントから、固定seed `20261010` で100件を抽出する。

| データセット | /m/ | /n/ | 合計 |
| --- | ---: | ---: | ---: |
| JVS | 25 | 25 | 50 |
| SRC4VC | 25 | 25 | 50 |
| 合計 | 50 | 50 | 100 |

各コーパス・音素の中で区間IDのseed付きSHA-256が小さい順に、重複なく選ぶ。
表示順は別のseedで混ぜる。話者・発話・長さ・音量による件数調整はしない。
最低長は現行入力の30msに合わせる。無音・低音量の除外、境界補正、正規化はしない。
元WAVのコピーは入力と同じバイト列で、単独区間WAVも元PCMの境界内だけを切り出す。
validation/testの話者・音声を抽出やレビューに使わない。

## 起動

リポジトリのルートで実行する。UIの既存Node.js環境と依存パッケージを使う。

```sh
python3 poc/phoneme-consonant-review/sample_review.py
sh poc/phoneme-consonant-review/start-review.sh
```

[http://127.0.0.1:5173/](http://127.0.0.1:5173/) を開く。
音声はローカルの `artifacts/phoneme-consonant-review/mn-100-20261010-v1/media/` に置く。
元発話は `utterances/`、切り出した100区間は `segments/` に置き、Gitへは追加しない。

## 確認方法

1. 対象音素を確認し、「候補Aだけ再生」で切り出した区間を聞く。
2. 「前後100msを含めて再生」や「原音声を再生」で境界を確認する。
3. 区間だけで何が聞こえるかと、学習データとして利用できるかを回答する。

対象は「ま」「な」の音節全体ではなく、その中の子音部分 `/m/`・`/n/`。
日本語の「ん」を表す `/N/` や、拗音の `/my/`・`/ny/` は含めない。
単独区間では判断できず、文脈付きで分かる場合は、聞こえ方の回答を「判断できない」としてよい。
前後100msはレビューの補助であり、切り出した区間には追加していない。

回答と現在位置はブラウザへ自動保存される。
上部の「JSONLへ保存」で回答済み項目を `data/review/review-records.jsonl` に追記する。
再起動・再抽出で人手回答ファイルを上書きしない。
今回は品質確認用の抽出で、レビュー結果を学習区間の個別選別や境界修正にはまだ使わない。

## 成果物

- `data/review/review-dataset.json`: UI用の100件・対象区間・設問。
- `data/review/sample.jsonl`: 元音声・境界・区間WAV・SHA-256・音量・隣接音素の監査記録。
- `data/review/sampling-summary.json`: 抽出母集団、内訳、長さ、話者数、再現用seed。
- `data/review/review-records.jsonl`: 人手回答。レビュー前は未作成。

元の学習・評価アーティファクトには書き込まない。UI上の候補は匿名ID `A` を使う。

## Common Voice日本語の追加レビュー

既存学習に使ったCommon Voice 17日本語trainの70話者から、同じ音素 `/m/`・`/n/` を各50件、計100件抽出する。
抽出seed・最低長30ms・音量で選別しない条件・再生方法・設問は先のレビューと共通。
Common Voiceは単独で100件、JVS・SRC4VCは各50件なので、コーパス間の比率を比較するときは件数差を考慮する。

```sh
python3 poc/phoneme-consonant-review/sample_common_voice.py
sh poc/phoneme-consonant-review/start-common-voice-review.sh
```

[http://127.0.0.1:5174/](http://127.0.0.1:5174/) を開く。
JVS・SRC4VCの5173番サーバーと並行して使える。
音声は `artifacts/phoneme-consonant-review/cv-mn-100-20261010-v1/media/`、
UIデータ・サンプル・抽出集計は `data/common-voice-review/` に置く。
人手回答は `data/common-voice-review/review-records.jsonl` に保存し、ブラウザの回答も別のデータセットIDで管理する。

## SRC4VCだけの再確認

最初のJVS・SRC4VC混合レビューから、SRC4VCの50件（`/m/`・`/n/` 各25件）だけを表示する。
再抽出せず、同じ区間・音声・順序・設問を引き継ぐ。

```sh
python3 poc/phoneme-consonant-review/build_src4vc_review.py
sh poc/phoneme-consonant-review/start-src4vc-review.sh
```

[http://127.0.0.1:5175/](http://127.0.0.1:5175/) を開く。
音声は最初のレビュー用のローカルコピーを使い、UIデータ・サンプル・抽出集計は `data/src4vc-review/` に置く。
再確認の回答は別データセットIDで管理し、`data/src4vc-review/review-records.jsonl` に保存する。
混合レビューとCommon Voiceレビューの回答には書き込まない。
