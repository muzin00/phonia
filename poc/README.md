# 音素ベース音声認証 PoC

## 1. 目的

本PoCでは、音声を音素ラベル付きの波形区間として扱い、同じ音素に含まれる話者固有の特徴を利用して本人認証が可能かを検証する。

音声全体から話者表現を生成する既存方式と比較し、音素単位で比較条件を揃える方式の有効性と課題を明らかにする。

## 2. PoCの位置づけ

本PoCは研究仮説を検証するための最小構成とする。最初から製品化を目指すのではなく、各フェーズの結果を確認しながら段階的に進める。

PoCの結果に基づき、次のフェーズへ進むか、方式を変更するかを判断する。

## 3. 対象範囲

最初の検証対象は、日本語の5母音とする。

```text
/a/ /i/ /u/ /e/ /o/
```

音素区間の抽出には既存モデルを利用し、抽出した元波形を話者認証用の入力データとして扱う。

## 4. 全体フロー

```text
音声・発話情報
      ↓
音素と波形区間の抽出
      ↓
データセットの構築と品質確認
      ↓
音素別の話者表現を学習
      ↓
ユーザープロファイルの作成
      ↓
音素別の照合とスコア統合
      ↓
認証性能の評価
      ↓
既存方式との比較
```

## 5. フェーズ構成

### Phase 1: 音素・波形区間の抽出

既存モデルを利用して、音声から対象音素とその時間区間を取得し、元波形との対応を保持する。機械検査と、波形・スペクトログラム・区間再生を組み合わせた支援レビューにより、後続処理で利用可能な母音区間を抽出できる方式を選定する。

詳細は [音素アライメント評価](phoneme-alignment-evaluation/README.md) を参照する。

### Phase 2: データセット構築と品質確認

採用したアライナーの抽出結果を、学習・評価に利用できる形へ整理し、音素ラベル、区間品質、話者・収録条件との対応を確認する。個別区間の人手選別や修正は行わず、検証済みの機械規則を全区間へ一律に適用する。

区間品質と支援レビューの結果は、採用方式の性能や失敗傾向を集計し、系統的な問題に
対する一律の採用規則を検証するために使用する。個別の学習例を選別する判定、信頼度
スコア、品質に応じたサンプル重み、学習順序の調整には使用しない。Phase 2レビューで
利用困難が確認された`near_silent`は、固定したRMS規則によりPhase 3入力から一律に
除外し、除外前後の件数を結果へ残す。

詳細は [音素別話者データセット](phoneme-speaker-dataset/README.md) を参照する。

### Phase 3: 音素別話者表現の学習

同じ音素における同一話者内の変動と話者間の違いを学習する共通モデルを構築する。

### Phase 4: ユーザー登録

学習済みの共通モデルを利用し、ユーザーごとの音素別プロファイルを作成する。

固定encoderによる登録・保存・読込を実装した。初期登録は5母音を各10区間以上そろえ、
母音別の代表値と使用区間数、モデル・前処理の識別情報を保存する。
詳細は[Phase 4ユーザー登録設計](phoneme-user-registration/README.md)を参照する。

### Phase 5: 本人照合

入力音声から得た音素別の特徴と登録済みプロファイルを比較し、複数の結果を統合して照合スコアを算出する。

保存済みプロファイルと同じ固定encoderを使う照合・結果保存・読込を実装した。
初期仕様は全5母音・照合側各1区間以上。区間cosineを母音内で平均し、5母音を等重みで統合する。
詳細は[Phase 5本人照合設計と実行方法](phoneme-verification/README.md)を参照する。
実データの確認は少数例の動作検証であり、認証性能の評価はPhase 6で行う。

### Phase 6: 評価

未知話者や異なる収録条件を含むデータで認証性能を評価し、音素ごとの特性も確認する。

初期JVS評価の設計・実装・性能測定を完了した。1発話を1照合入力とし、validationで決めた閾値を
testへ固定して適用した。入力不足も件数と拒否率に残す。登録各母音10区間の主条件では、
testの他人受入率0.729%、条件付き本人拒否率2.585%、全入力本人拒否率4.533%、統合EER 1.526%だった。
JVSでは別発話での性能を測り、別日・別端末での性能は追加データによる後続評価とする。
詳細は[Phase 6認証性能評価設計](phoneme-verification-evaluation/README.md)を参照する。
全条件・信頼区間と限界は[Phase 6評価結果](phoneme-verification-evaluation/evaluation-results.md)へ記録した。

### Phase 7: ベースライン比較

設計・実装・validation pilot・全validation・実行凍結・test・全条件レポートを完了した。
SpeechBrain ECAPA-TDNN、境界内母音、前後各20 msの文脈付き母音を、同じ登録元WAV・照合発話で比較した。
登録1/5/10、fullと最大1/2/3/5秒、通常文/別テキスト、native/commonの全360条件を報告した。
validation/testで計1,620,000 score slotsを生成し、validation通常文から決めた270閾値をtestへ固定適用した。
10,000回の話者bootstrap CI、792個のpaired差、元発話長と実利用時間の層、60図のPNG/PDFを保存した。

主条件のtest全入力FRRは、境界内母音4.533%、文脈付き母音4.533%、ECAPA 0.000%。
ECAPA−境界内の差は−4.533 points、95% CI [−8.933, −2.000]だった。
ECAPAが優位だったが、学習・モデル規模・内部利用音声量も異なり、音素分割の因果効果は評価していない。
別日/別端末での性能と独立した新holdoutは後続研究の対象となる。

[比較設計](phoneme-baseline-comparison/comparison-design.md)、[再現手順](phoneme-baseline-comparison/README.md)、
[結果と限界](phoneme-baseline-comparison/evaluation-results.md)、[全条件HTML](phoneme-baseline-comparison/evaluation-results.html)を参照する。
[モデル選定](phoneme-baseline-comparison/model-selection.md)、[pilot](phoneme-baseline-comparison/validation-pilot-results.md)、
[全validation](phoneme-baseline-comparison/validation-results.md)も保持する。
30 msのECAPA登録入力が失敗した[短入力診断](phoneme-baseline-comparison/short-input-results.md)を踏まえ、
元WAVごとの内部利用時間合わせを撤回し、共通取得波形の長さ上限を比較する設計へ改訂した。

### Phase 8: 音素方式の改善

最初の検証として、境界内母音の登録区間数を各母音10・20・30へ増やす実測を完了した。
固定encoder・同一の発話全区間で照合し、登録数ごとのvalidation閾値をtestへ固定適用した。
validation/test計108,000 trial、10,000回の共有話者bootstrap、72個のpaired差を保存した。

目標FAR 1%では10→30で全入力FRRは通常文4.533%、別テキスト14.667%のまま、FAR/EERの観測値は低下した。
目標FAR 0.1%では全入力FRRは通常文14.533→12.400%、別テキスト28.444→24.222%へ下がったが、実測FARは少し上がった。
これは既に観測済みのJVS testを使う探索的な追加検証。

[固定条件・再現手順](phoneme-enrollment-scaling/README.md)、[測定結果](phoneme-enrollment-scaling/evaluation-results.md)、
[結果の解釈](phoneme-enrollment-scaling/interpretation.md)、[HTML比較表](phoneme-enrollment-scaling/evaluation-results.html)を参照する。

2番目の検証として、登録各母音10区間を固定し、照合の5母音必須・4母音以上・3母音以上を実測した。
4母音以上では主条件の全入力FRRが通常文4.533→2.533%、別テキスト14.667→3.333%へ低下した。
母音不足の本人発話を通常文15件・別テキスト51件救済し、他人受入は通常文75→77件、別テキスト46→49件へ増えた。
3母音以上では別テキスト全入力FRRが2.444%まで下がったが、3母音のtest入力は5発話しかない。
生score、10,000回話者bootstrap、本人救済・既存判定の変化と欠損パターン別の内訳を保存した。

[母音不足対応の固定条件・再現手順](phoneme-missing-vowel-evaluation/README.md)、
[測定表](phoneme-missing-vowel-evaluation/evaluation-results.md)、[結果の解釈](phoneme-missing-vowel-evaluation/interpretation.md)、
[HTML比較表](phoneme-missing-vowel-evaluation/evaluation-results.html)を参照する。
この検証も観測済みJVS testの探索的な追加検証。

学習データ増加は、現行JVS 70話者と、JVS＋Common Voice日本語70話者ラベルの2条件・1 seedに絞った。
固定した65,920 parametersのencoderを初期化から追加条件で1 run学習し、18,000 updateで早期終了した。
学習区間は323,913→522,359件。15,000 updateのcheckpointを採用し、推論用重みの再読込も検証した。
母音単一区間のvalidation macro EERは9.196→9.274%で、今回のvalidationでは改善を確認できなかった。

追加データで話者ラベル数・区間数・収録環境が同時に変わる。匿名client_idとJVS評価話者の実在人物の重複は確認できない。
testは学習・checkpoint選択に使用せず、固定した2モデルの発話単位評価まで完了した。
登録各母音10区間・5母音必須、validation目標FAR 1%の閾値で全入力FRRは通常文4.533→3.067%、別テキスト14.667→13.556%へ下がった。
全入力FARは0.714→2.629%、0.730→1.905%へ上がり、FARを維持した精度改善は確認できなかった。
主条件の対応付き差の95% CIはいずれも0を含む。観測済みJVS testの探索的な結果で、母音不足対応との組み合わせ・子音追加・独立holdoutは未評価。

[学習データ追加の固定条件・再現手順](phoneme-training-data-expansion/README.md)、
[学習結果](phoneme-training-data-expansion/training-results.md)、[HTML比較表](phoneme-training-data-expansion/training-results.html)を参照する。
[発話単位の評価表](phoneme-training-data-expansion/evaluation-results.md)、
[発話単位HTML](phoneme-training-data-expansion/evaluation-results.html)、[評価の解釈](phoneme-training-data-expansion/evaluation-interpretation.md)を参照する。

## 6. 基本方針

- 音素方式が既存方式より優れているとは仮定しない。
- 各フェーズを再現可能な形で実施する。
- 学習用データと評価用データの混入を防ぐ。
- 新規ユーザーの追加時に共通モデルを再学習しない。
- 類似度と本人である確率を区別する。
- 各フェーズの品質を確認してから次へ進む。

## 7. PoCで扱わない範囲

初期PoCでは、以下を対象外とする。

- 本番利用者向けのWeb UIおよびモバイルUI（研究用のローカル支援レビューUIは対象に含む）
- 学習区間の人手選別、境界修正、再アノテーション
- 区間ごとの信頼度推定、品質スコアの合成、信頼度に基づく除外や重み付け
- 本番環境向けの認証基盤
- 全日本語音素への対応
- 継続的なオンライン学習
- なりすまし音声や生成音声への包括的な対策
- 大規模な分散処理基盤

## 8. 詳細設計

各フェーズで使用するモデル、データ形式、学習方式、評価条件などの具体的な内容は、フェーズごとの設計時に別途定義する。
