# Phonia

Phoniaは、音素単位の話者特徴を利用した音声本人認証の可能性を検証する研究プロジェクトです。

## 概要

一般的な話者認証では、発話全体から一つのspeaker embeddingを生成し、登録済みの特徴と比較します。

本プロジェクトでは、音声を音素ラベル付きの波形区間へ分割し、同じ音素の中で「同一話者の揺らぎ」と「話者間の違い」を学習します。音素によって比較条件を揃えることで、発話内容による変動を抑えながら、話者固有の特徴をより安定して捉えられるかを検証します。

```text
録音音声 + 発話テキスト
          ↓
既存モデルによる音素アライメント
          ↓
音素ラベル + 対応する元波形区間
          ↓
音素別の話者表現
          ↓
登録済みプロファイルとの比較・統合
          ↓
本人照合スコア
```

共通の音声モデルは多数話者のデータで学習し、新規ユーザーの追加時には再学習せず、ユーザーごとの音素別プロファイルを登録します。

本プロジェクトは研究PoCの段階です。音素ベースの方式が既存方式より優れているとは仮定せず、同一条件で比較して有効性と限界を確認します。

## 現在の状態

最初の検証対象は日本語の5母音 `/a/ /i/ /u/ /e/ /o/` です。Phase 1ではJuliusを母音区間の抽出方式に採用し、Phase 2ではJVS 100話者からPhase 3用の母音区間462,242件を構築しました。

Phase 3では共通encoderの学習と固定validation trialでの登録・照合評価を実装し、18設定×3 seedの54 runを完了しました。事前に決めた規則でlog-Mel・統計pooling + MLP・RMS正規化なし・AAM-Softmax + 母音内SupConを採用し、validationの5母音平均EERは3 seed平均で9.436%でした。

採用構成の学習話者数別の評価と境界ずれ・音量変化への感度診断、設定と評価手順を凍結した最終test評価まで完了しました。testの5母音平均EERは3 seed平均で10.028%でした。

Phase 4では固定encoderからユーザー別の母音プロファイルを作成し、保存・読込する処理を実装しました。Phase 5では同じ母音のcosine similarityを母音内で平均し、5母音を等重みで統合する照合処理を実装しました。保存済みプロファイルと別のvalidation音声で、スコア出力・再現性・モデルとプロファイルの不変を確認しています。

Phase 6では、1発話を1入力とする照合をvalidation/test各15話者で評価しました。登録各母音10区間の主条件では、validationでFAR 1%を目標に決めた閾値を固定してtestへ適用し、他人受入率0.729%、本人拒否率2.585%、入力不足を含む本人拒否率4.533%、統合scoreのEER 1.526%でした。JVSの統制収録内での結果であり、別日・別端末での性能は未確認です。Phase 7の比較も完了しました。

Phase 7では、境界内母音・前後20 msの文脈付き母音・発話全体ECAPAを、同じ登録元WAVと照合発話で比較しました。実行条件とvalidation閾値を凍結し、testの全条件、話者信頼区間、方式差、長さ別の性能と入力不足率を評価しました。主条件の全入力本人拒否率はECAPA 0.000%、母音2方式は4.533%でした。ECAPAが優位でしたが、モデル規模・事前学習・内部の利用音声量も異なるため、音素分割だけの効果は切り分けられません。

## ドキュメント

- [PoC概要](poc/README.md)
- [研究設計](docs/research-design.md)
- [Phase 3: 音素別話者encoderの学習・評価](poc/phoneme-speaker-encoder/README.md)
- [Phase 3: 70話者・3 seed比較結果](poc/phoneme-speaker-encoder/phase3-70spk-comparison-results.md)
- [Phase 3: 最終test評価結果](poc/phoneme-speaker-encoder/phase3-final-test-results.md)
- [Phase 4: ユーザー登録設計と実行方法](poc/phoneme-user-registration/README.md)
- [Phase 5: 本人照合の設計と実行方法](poc/phoneme-verification/README.md)
- [Phase 5: 実データの動作確認結果](poc/phoneme-verification/smoke-results.md)
- [Phase 6: 本人照合の認証性能評価設計](poc/phoneme-verification-evaluation/README.md)
- [Phase 6: JVS発話単位の認証性能評価結果](poc/phoneme-verification-evaluation/evaluation-results.md)
- [Phase 6: 評価結果のブラウザ用HTML](poc/phoneme-verification-evaluation/evaluation-results.html)

- [Phase 7: ベースライン比較結果・信頼区間・限界](poc/phoneme-baseline-comparison/evaluation-results.md)
- [Phase 7: 全条件表と図表のHTML](poc/phoneme-baseline-comparison/evaluation-results.html)
