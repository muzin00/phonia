# 5母音＋ /m/・/n/ の共通encoder学習

JVS70話者とCommon Voice日本語70話者で、`/a/ /i/ /u/ /e/ /o/ /m/ /n/` の共通encoderを1条件・1 seedで学習する。
SRC4VCは、[事前の聴取所見](../phoneme-consonant-review/README.md)を受けて今回の学習から外す。

## 固定条件

| 項目 | 設定 |
| --- | --- |
| encoder | statistics pooling＋MLP、65,920 parameters、128次元 |
| 初期化 | 初期値から学習、seed 20260926 |
| 学習量 | 30,000更新、早期終了なし、最後の重みを採用 |
| 入力 | アライメント境界内、24kHz mono PCM16、30〜250ms。長い区間は学習時ランダムcrop |
| 新規子音の除外 | 30ms未満、無音またはRMSが−50dBFS未満 |
| 前処理 | DC除去、RMS正規化なし、64-bin log-Mel、trainだけの特徴統計 |
| 損失 | AAM-Softmax＋同じ音素内SupCon（重み0.5） |
| optimizer | 既存Phase 3と同じAdamW、warmup 1,000＋cosine decay |
| 1更新 | 10話者×5音素×各2区間＝100区間、5個の完全な20区間グループ |
| 音素選択 | 7音素から連続5音素を選び、更新ごとに開始位置を5進める。7更新で各音素5回 |
| 区間選択 | 話者・音素・更新・seedに基づく重複なし抽出 |
| 話者分割 | 既存trainを使用、JVS validation15/test15は学習に含めない |

`/N/`（撥音）、`/my/`・`/ny/` は `/m/`・`/n/` にまとめない。
既存の5母音用コード・manifest・凍結済み実験は変更せず、新しいサンプラーから共通処理を利用する。
既存コードの `Segment.vowel` とバッチの `vowels` は音素ラベルを保持するフィールドとして再利用する。

母音は既存JVS＋CVの522,359区間とtrain特徴統計をそのまま再利用する。
追加子音のtrain統計を別途計算し、母音との母集団モーメントを結合する。
140話者はコーパス内の話者ラベル数であり、コーパス間の実在人物の重複は未確認。

## 人間レビューの採用記録

ユーザーは事前にJVSを「品質はとても良い」、Common Voiceを「OK」と判断し、
その後「\"/m /n\"の学習作業進めて」と指示した。この所見と指示を今回の採用根拠として記録する。
`adoption.json` にレビューサンプルの自動チェック後の残存数も保存する。

[通常の採用手順](../../docs/training-data-adoption.md)に対し、今回はユーザーの明示的な学習開始指示に基づいて進める。
新たな自動除外後の100件／コーパスのレビューと個別回答の集計は行っていない。
事前サンプルの件数を回答済み件数と扱わず、聴感上の無音率・誤音率・利用可能率も推定しない。
母音は今回の子音予備レビューの対象外で、既存の採用データを継続使用する。

## 実行

リポジトリrootから、既存の機械学習venvを使用する。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-consonant-training/tests -v
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-training/train.py prepare
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-training/train.py freeze
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-training/train.py train
```

設定は [protocol.json](config/protocol.json)。生成先は
`artifacts/phoneme-consonant-training/jvs-cv-vowels-mn-20261010-v1/`。
新規子音の除外前・除外後・除外理由別manifest、話者／音素数、採用理由、元WAV・アライメントのSHA-256を残す。
学習前後に入力・コードのchecksumを照合し、checkpointから同じサンプル列とRNG状態で再開できる。

最終bundleは7音素を入力できる共通encoderで、学習時の話者分類headは登録・照合には使わない。
7音素それぞれについてexport前後の128次元出力が完全一致することを確認する。

2026-10-10に30,000更新の学習を完了した。[学習結果とvalidation診断](training-results.md)を参照する。

## 今回のvalidationの意味

5,000更新ごとに既存の固定JVS validationで**母音単一区間のmacro EER**を診断する。
この値はcheckpoint選択や早期終了に使わず、最後の30,000更新を採用する。
testは使わない。この診断だけでは、子音追加で発話単位FAR・FRR・EERが改善したとは判断できない。

後続の比較では、同じ学習済み共通encoderを使い、5母音のみと5母音＋m/nを比較する。
登録・照合それぞれで実際に使用したPCM秒数を揃え、子音を単に追加して入力量が増える条件とは区別する。
対象話者・入力集合、音素不足率、validationでの閾値校正を固定してからtestを評価する。
2026-10-10にこの比較のvalidation/test実測まで完了した。
[使用音声量を揃えた比較と再現手順](../phoneme-consonant-evaluation/README.md)、
[HTML比較表](../phoneme-consonant-evaluation/evaluation-results.html)、
[結果の解釈](../phoneme-consonant-evaluation/interpretation.md)を参照する。
