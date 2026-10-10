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

Phase 8の最初の検証として、固定した境界内母音encoderで各母音の登録10・20・30区間を実測しました。目標FAR 1%では10→30で本人拒否率は変わらず、FAR/EERの観測値は低下しました。目標FAR 0.1%では本人拒否率が下がる一方、実測FARは少し上がりました。既に観測したJVS testを使う探索的な追加検証であり、Phase 8の残りの改善と独立holdout評価は未実施です。

Phase 8の2番目の検証では、登録各母音10区間を固定し、照合の5母音必須・4母音以上・3母音以上を比較しました。4母音以上では主条件の全入力本人拒否率が通常文4.533→2.533%、別テキスト14.667→3.333%へ下がりました。全入力他人受入率は通常文0.714→0.733%、別テキスト0.730→0.778%へ上がり、FARの維持は証明していません。観測済みJVS testでの探索的な結果です。

Phase 8の学習データ増加では、現行JVSと、Common Voice日本語の追加70話者ラベルを含む2条件・1 seedに絞り、追加学習1 runと発話単位評価を完了しました。学習区間は323,913→522,359件、encoderは65,920 parametersで固定しました。18,000 updateで早期終了し、15,000 updateのcheckpointを採用しました。母音単一区間のvalidation macro EERは9.196→9.274%でした。登録各母音10区間・5母音必須の発話単位評価では、validation目標FAR 1%で全入力FRRが通常文4.533→3.067%、別テキスト14.667→13.556%へ下がる一方、全入力FARは0.714→2.629%、0.730→1.905%へ上がりました。今回の探索的な評価では総合的な精度改善を確認できませんでした。子音追加・独立holdoutは未評価です。

続く切り分けでは、同じ追加データモデルを30,000 updateまで延長し、JVS各話者のbatch選択回数を現行モデルの2,142〜2,143回に揃えました。新規条件は1つ・1 seed。全入力FARは通常文2.629→2.657%、別テキスト1.905→2.000%で、学習延長によるFAR改善は確認できませんでした。全入力FRRは通常文で1件分の3.067→2.933%、別テキスト13.556→13.556%でした。固定した学習率scheduleでの探索的な結果であり、最適な学習予算・収録条件・モデル容量は未切り分けです。

別コーパスSRC4VCの原録音から追加70話者を選び、JVS70＋SRC4VC70の1条件・1 seedで30,000 updateの学習を完了しました。JVSを含む学習可能な母音は415,271区間です。同じ30,000 updateのCommon Voice版とのvalidation macro EER比較は9.228→8.887%でした。データ量・話し方・収録条件・特徴統計も変わる探索的な比較で、属性の偏りだけを原因とする結果ではありません。後続の発話単位評価まで完了し、CV版との全入力FAR比較は通常文2.657→1.714%、別テキスト2.000→1.413%でした。全入力FRRは通常文2.933→2.400%、別テキスト13.556→13.778%。主比較の差の95% CIはいずれも0を含み、明確な優位性は確認できませんでした。

5母音＋ /m/・/n/ の学習では、聴取所見を受けてJVS＋Common Voiceを採用し、SRC4VCを外しました。140話者ラベル・594,630区間で30,000更新を完了しました。同じ7音素encoderを使い、登録・照合の実使用音声量を揃えた比較まで実施しました。全7音素が使える通常文525件では、5母音→m/n追加でFAR 1.578→1.007%、FRR 4.571→2.095%、EER 2.395→1.333%でした。別テキスト265件も3指標の観測値が下がりました。主比較では通常文FRRの差の95% CIのみが0を含まず、観測済みJVS test・1 seedでの探索的な改善の兆候です。

## ドキュメント

- [PoC概要](poc/README.md)
- [研究設計](docs/research-design.md)
- [学習データの採用手順：自動チェック・人間レビュー・採用判断](docs/training-data-adoption.md)
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
- [Phase 8: 登録10・20・30区間の実測と解釈](poc/phoneme-enrollment-scaling/interpretation.md)
- [Phase 8: 登録区間数のHTML比較表](poc/phoneme-enrollment-scaling/evaluation-results.html)
- [Phase 8: 照合時の母音不足への対応と実測の解釈](poc/phoneme-missing-vowel-evaluation/interpretation.md)
- [Phase 8: 母音不足対応のHTML比較表](poc/phoneme-missing-vowel-evaluation/evaluation-results.html)
- [Phase 8: 学習データ追加の固定条件と再現手順](poc/phoneme-training-data-expansion/README.md)
- [Phase 8: 学習データ追加の学習結果HTML](poc/phoneme-training-data-expansion/training-results.html)
- [Phase 8: 学習データ追加の発話単位評価HTML](poc/phoneme-training-data-expansion/evaluation-results.html)
- [Phase 8: 学習データ追加の評価結果の解釈](poc/phoneme-training-data-expansion/evaluation-interpretation.md)
- [Phase 8: 学習回数を揃えた切り分けのHTML比較表](poc/phoneme-training-exposure-ablation/evaluation-results.html)
- [Phase 8: 学習回数の切り分けと残る仮説](poc/phoneme-training-exposure-ablation/interpretation.md)
- [Phase 8: SRC4VCでの学習条件と再現手順](poc/phoneme-src4vc-training/README.md)
- [Phase 8: SRC4VCの学習結果HTML比較表](poc/phoneme-src4vc-training/training-results.html)
- [Phase 8: SRC4VCの発話単位評価HTML](poc/phoneme-src4vc-evaluation/evaluation-results.html)
- [Phase 8: SRC4VC評価の解釈](poc/phoneme-src4vc-evaluation/interpretation.md)
- [Phase 8: 30,000・45,000・60,000回の学習量比較の固定条件](poc/phoneme-training-budget/README.md)
- [Phase 8: 60,000回までの学習曲線HTML](poc/phoneme-training-budget/training-results.html)
- [Phase 8: 30,000・45,000・60,000回の発話単位評価HTML](poc/phoneme-training-budget/evaluation-results.html)
- [Phase 8: 学習回数の比較結果と解釈](poc/phoneme-training-budget/interpretation.md)
- [Phase 8: 5母音＋ /m/・/n/ の共通encoder学習](poc/phoneme-consonant-training/README.md)
- [Phase 8: 5母音＋ /m/・/n/ の学習結果とvalidation診断](poc/phoneme-consonant-training/training-results.md)
- [Phase 8: /m/・/n/ 追加の使用音声量を揃えたHTML比較表](poc/phoneme-consonant-evaluation/evaluation-results.html)
- [Phase 8: /m/・/n/ 追加の評価結果と解釈](poc/phoneme-consonant-evaluation/interpretation.md)
