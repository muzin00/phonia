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

[Issue #21](https://github.com/muzin00/phonia/issues/21)で学習・評価仕様を設計した。
入力方式は事前に一つへ固定せず、可変長log-Mel、生波形などの候補を実際に学習し、
未知話者のvalidation照合性能から選択する。

入力データ、候補となる入力特徴、可変長処理、encoder、損失関数、batch sampling、成果物形式、
validationとtestの用途を設計済みである。判断が難しい項目は互換性のある候補を組み合わせ、
未知話者のvalidation照合性能から採用する。設計version 2.0.0では、本比較はlog-Mel 8通り・
生波形8通りの計16通りを維持し、規模を揃えた時間混合なしモデルと長文脈生波形モデルを
各1設定の限定比較として追加する。全18設定が採用対象である。

過学習確認6、成立性確認18、70話者比較54、採用構成の学習曲線追加9の計87学習runを計画する。
品質層・境界ずれ・音量への診断は再学習せず行う。採用規則を事前固定し、選択した構成の
3 seedすべてを凍結してtestを一度評価する。固定した128次元等の最適性や、入力方式全般の
優劣を保証する設計ではない。既存全発話方式との比較はPhase 7、実環境は外部評価段階で扱う。

次はDataset / DataLoaderと特徴統計を実装する。資源上限・実行環境は学習結果を見る前に
登録し、設計の検証と学習実装のテストを通してから比較を開始する。学習結果はまだない。

## 4. ドキュメント

- [入力設計](input-design.md)
- [生波形encoder設計](waveform-encoder-design.md)
- [学習・評価設計](training-evaluation-design.md)
- [候補比較計画](comparison-plan.md)
- [参照log-Mel設定](config/baseline-log-mel.json)
- [log-Mel encoder設定](config/log-mel-encoders.json)
- [探索空間](config/search-space.json)
- [共通実験プロトコル](config/experiment-protocol.json)
- [生波形encoder設定](config/waveform-encoders.json)
- [研究全体の設計](../../docs/research-design.md)
- [Phase 2音素別話者データセット](../phoneme-speaker-dataset/README.md)

## 5. 設計設定の整合性検証

リポジトリrootで次を実行する。標準ライブラリだけでJSON参照、候補・run数、parameter数、
受容野・出力長、batch構成、seedと共有設定の整合性を検査する。学習実装や実データでの検証ではない。

```sh
python3 -m unittest discover -s poc/phoneme-speaker-encoder/tests -v
```
