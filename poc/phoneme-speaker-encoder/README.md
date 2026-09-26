# 音素別話者encoder

## 1. 目的

Phase 3では、母音の短い波形区間から固定長の話者embeddingを生成する共通encoderを
学習し、学習に含まれない話者の登録音声と照合音声でも話者差を捉えられるか検証する。

同じ母音という条件の中で、次の関係を学習目標とする。

```text
同じ話者・同じ母音   → embedding空間で近づける
異なる話者・同じ母音 → embedding空間で遠ざける
```

## 2. 対象範囲

対象は、日本語の5母音 `/a/ /i/ /u/ /e/ /o/` の1区間を入力とする共通encoderである。
共通encoderはtrain話者で学習し、validation、test、新規ユーザーの登録時には更新しない。

複数母音のembeddingやスコアをTransformerで統合する処理は、
[Issue #22](https://github.com/muzin00/phonia/issues/22)で扱う。

## 3. 現在の状態

[Issue #21](https://github.com/muzin00/phonia/issues/21)で学習・評価仕様を設計中である。
入力方式は事前に一つへ固定せず、可変長log-Mel、生波形などの候補を実際に学習し、
未知話者のvalidation照合性能から選択する。

入力データ、話者分割、validationとtestの用途は確定済みである。encoder構造、損失関数、
batch sampling、入力特徴の詳細値は未決定である。

## 4. ドキュメント

- [入力設計](input-design.md)
- [学習・評価設計](training-evaluation-design.md)
- [研究全体の設計](../../docs/research-design.md)
- [Phase 2音素別話者データセット](../phoneme-speaker-dataset/README.md)
