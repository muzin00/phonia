# 母音5種＋m/n/sの共通encoder学習

子音の組み合わせを比較するため、JVS70話者＋Common Voice日本語70話者で
`a i u e o m n s` の8音素を入力するモデルを1条件・1 seedで学習する。
SRC4VCは含めない。既存の7音素モデルや凍結済み実験は変更しない。

| 項目 | 固定条件 |
| --- | --- |
| encoder | statistics pooling＋MLP、65,920 parameters、128次元 |
| 初期化 | 初期値から学習、seed 20260926 |
| 学習量 | 30,000更新、早期終了なし、最後の重みを採用 |
| データ | JVS・Common Voiceのtrain各70話者、621,472区間 |
| 追加/s/ | 26,842区間。準備済みの自動チェック後eligible manifestをそのまま使用 |
| 特徴統計 | 既存7音素のtrain統計と新規/s/のtrain統計を母集団モーメントで結合 |
| 入力 | 24kHz mono PCM16、DC除去、RMS正規化なし、64-bin log-Mel |
| 損失・optimizer | 従来と同じAAM-Softmax＋SupCon、AdamW、warmup 1,000＋cosine decay |
| 1更新 | 10話者×5音素×各2区間＝100区間 |
| 音素選択 | 8音素から連続5音素、開始位置を更新ごとに5進める。8更新で各音素5回 |
| 診断 | 5,000更新ごとに固定validationの母音単一区間macro EER。重み選択には使わない |

母音・m/nは[既存の学習](../phoneme-consonant-training/README.md)の入力を再利用する。
/s/は[準備済み候補](../phoneme-s-review/README.md)と同じく、元音素境界内で最大250msにcenter cropした後に
30ms以上・RMS−50dBFS以上を確認した区間を使用する。波形に文脈や無音padを加えない。
母音・m/nの長い区間は従来どおり学習時にランダムcropする。
従って、全8音素に新しい/s/の切り出し規則を適用し直した実験ではない。

通常の試聴・検証として進め、今回の個別回答保存・100件の本格的レビュー・聴感率の集計は行わない。
[採用手順](../../docs/training-data-adoption.md)の本格的レビューはユーザーが明示した場合に実施する。
再現に必要な入力manifest、設定、元音声・コードのchecksumを固定する。
140話者はコーパス内の話者ラベル数で、コーパス間の実在人物の重複は未確認。

## 実行

リポジトリrootから実行する。

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-consonant-combination-training/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-training/train.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-training/train.py freeze
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-training/train.py train
```

設定は[protocol.json](config/protocol.json)。出力先は
`artifacts/phoneme-consonant-combination-training/jvs-cv-vowels-mns-20261010-v1/`。
既存の入力と追加/s/のchecksumを準備・学習前後に検証する。
checkpointにはoptimizer・サンプラー・RNG状態を保存し、同じ区間列で再開する。
export前後の出力が8音素すべてで完全一致することも確認する。

発話照合の改善は母音単一区間の診断だけで判定せず、
[同じモデル・同じ音声量での比較](../phoneme-consonant-combination-evaluation/README.md)で評価する。

2026-10-10に30,000更新の学習とvalidation/testの比較を完了した。
[学習結果](training-results.md)・[子音組み合わせの評価と解釈](../phoneme-consonant-combination-evaluation/interpretation.md)を参照する。
