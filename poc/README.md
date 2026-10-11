# 音素ベース音声認証 PoC

初期PoCと主要な改善探索を完了した。[最終結論と次段階](conclusion.md)では、本人照合の技術的な成立、改善する条件、既存方式への優位性が未確認であることをまとめている。次段階は構成を固定したプロトタイプと新しい実録音での評価とする。

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

学習機会の減少を切り分けるため、追加データモデルを15,000→30,000 updateまで同じ状態から延長した。
JVS各話者のbatch選択回数は現行モデルと同じ2,142～2,143回、各話者の差は最大1回だった。
データ・特徴統計・encoder・損失・batch・学習率scheduleを固定し、新規条件は1つ・1 seedに限定した。
validation目標FAR 1%で全入力FARは通常文2.629→2.657%、別テキスト1.905→2.000%となり、改善は確認できなかった。
全入力FRRは通常文3.067→2.933%、別テキスト13.556→13.556%。通常文で本人1件を救済した。
学習機会半減だけが原因という仮説は弱まったが、最適な学習予算・学習率やコーパス・モデル容量の影響は未検証。
[固定条件と再現手順](phoneme-training-exposure-ablation/README.md)、[HTML比較表](phoneme-training-exposure-ablation/evaluation-results.html)、
[切り分けの解釈](phoneme-training-exposure-ablation/interpretation.md)を参照する。

別コーパスのSRC4VCでは、歌唱・復元音声を除いた原録音を使用した。
100話者を固定hashで70/15/15に分割し、予約30話者の音声は区間抽出・特徴統計・学習・validationに使わなかった。
train 70話者・3,500発話のうち3,499発話を抽出でき、追加91,358区間、JVSを含む合計415,271区間となった。
同一初期化・encoder・scheduleの1条件・1 seedを30,000 updateまで学習し、JVS validationを完了した。
同じ採用30,000 updateのCommon Voice版とのvalidation macro EER比較は9.228→8.887%。
データ量・話し方・収録条件・特徴統計が同時に変わる探索的な観測である。
続く発話単位test評価では、CV版との全入力FAR比較が通常文2.657→1.714%、別テキスト2.000→1.413%となった。
全入力FRRは通常文2.933→2.400%、別テキスト13.556→13.778%。主比較の差の95% CIはいずれも0を含む。
現行JVSモデルよりはFARが高く、総合的な優位性を確認した結果ではない。
[発話単位HTML比較表](phoneme-src4vc-evaluation/evaluation-results.html)、[結果の解釈](phoneme-src4vc-evaluation/interpretation.md)を参照する。
[固定条件と再現手順](phoneme-src4vc-training/README.md)、[学習結果HTML](phoneme-src4vc-training/training-results.html)を参照する。

続く学習予算検証では、Common Voice版とSRC4VC版を初期化から各60,000 updateまで学習し、
同じ60,000回用schedule内の30,000・45,000・60,000回checkpointを比較する。
新規学習は2本・各1 seed。データと特徴統計を固定し、checkpointとvalidation閾値をtestで選び直さない。
学習と6条件の発話単位評価まで完了した。Common Voiceのtest FARは通常文2.305→2.038%、別テキスト1.794→1.571%と低下した。
SRC4VCの通常文EERは1.224→1.098→0.962%と低下したが、別テキストFRRは13.778→14.222%に増えた。
30,000→60,000回の主比較12指標の差の95% CIは全て0を含み、総合的な改善は確認できていない。
単一母音区間のvalidation EERは30,000回以降ほぼ横ばいだった。
[HTML比較表](phoneme-training-budget/evaluation-results.html)、[結果の解釈](phoneme-training-budget/interpretation.md)、
[学習曲線](phoneme-training-budget/training-results.html)を参照する。
[固定条件と再現手順](phoneme-training-budget/README.md)を参照する。

子音追加の学習前に、JVS・SRC4VCのtrain音声から /m/・/n/ を各25件、計100件抽出した。
[音素区間レビューの起動・確認方法](phoneme-consonant-review/README.md)を参照する。
既存レビューUIで単独区間と文脈付き音声を確認し、人手回答を専用JSONLに保存する。
Common Voice日本語trainの /m/・/n/ 各50件を確認する追加レビューも別ポート・別回答ファイルで用意した。

聴取所見を受け、次の子音追加ではJVS＋Common Voiceを使用し、SRC4VCを外す。
1条件・1 seedで5母音＋ /m/・/n/ の共通encoderを学習する。
[固定条件と再現手順](phoneme-consonant-training/README.md)を参照する。
140話者ラベル・594,630区間で30,000更新を完了し、全7音素でexport前後の出力一致を確認した。
最終更新の母音単一区間validation macro EERは9.716%。
[学習結果と診断表](phoneme-consonant-training/training-results.md)を参照する。

同じ7音素encoderで5母音のみとm/n追加の発話単位比較まで完了した。
登録最大3秒・照合最大1秒の実使用PCMサンプル数を方式間で揃え、validation通常文の閾値をtestへ固定した。
全7音素が使える通常文525件ではFAR 1.578→1.007%、FRR 4.571→2.095%、EER 2.395→1.333%。
別テキスト265件でも3指標の観測値は低下した。主比較では通常文FRRの差の95% CIのみが0を含まなかった。
全発話を含む補助比較では照合可能数は変わらず、通常文の全入力FRRは6.533→4.267%、別テキストは14.222→13.778%。
1 seed・観測済みJVS testの探索的な比較で、従来の5母音学習モデルや独立holdoutとの比較ではない。
[比較の固定条件](phoneme-consonant-evaluation/README.md)、[HTML比較表](phoneme-consonant-evaluation/evaluation-results.html)、
[測定表](phoneme-consonant-evaluation/evaluation-results.md)、[結果の解釈](phoneme-consonant-evaluation/interpretation.md)を参照する。

5母音を固定し、JVS・Common Voiceに存在する追加32音素をランダム順で試す探索を完了した。
改善した音素とencoderを保持し、採用セット変更後に不採用候補を再試験した。
3巡・87回の採否、基準を含む69モデルの30,000更新学習と最終test評価を実施した。
追加31音素を実測し、学習区間が0件のtyはデータ不足として候補に残した。
最終セットは5母音＋b・gy・ry・s・v。validation通常文EERは0.671→0.447%だったが、
test通常文は1.924→2.177%、別テキストは1.859→2.278%となり、testでの改善は確認できなかった。
全モデルで共通の評価発話を使い、採否にtestは使用していない。1,278,000スコアの再計算チェックを完了した。
[固定条件と再現手順](phoneme-greedy-selection/README.md)、[HTML比較表](phoneme-greedy-selection/evaluation-results.html)、
[全試行の測定表](phoneme-greedy-selection/evaluation-results.md)、[結果の解釈](phoneme-greedy-selection/interpretation.md)を参照する。

続いて、EERによる音素選択をせず、学習可能な全36音素を1モデルで学習・評価した。
新規学習は1本・1 seed、216,000更新。基準5母音モデルと各母音への延べ投入を600,000区間に揃えた。
同じ入力と基準モデルを再利用し、最終更新の重みとvalidation閾値を固定してtestを測定した。
test通常文EERは1.924→1.769%、別テキストは1.859→1.531%と観測値が低下した。
対応付き差の95% CIは両方とも0を含み、改善の確証には至っていない。
総学習更新と追加音素の利用音声量も増える構成の比較で、音素数だけの因果効果は測っていない。
tyは適合学習区間0件で未学習として記録した。
[固定条件と再現手順](phoneme-all-training/README.md)、[HTML比較表](phoneme-all-training/evaluation-results.html)、
[測定表](phoneme-all-training/evaluation-results.md)、[結果の解釈](phoneme-all-training/interpretation.md)を参照する。

全36音素の学習済みencoderを固定して、登録区間数の上限10・20・30も実測した。
モデルを再学習せず、音素・照合発話を共通にし、登録10の区間を20・30にも含めた。
希少音素は実在区間だけを使い、各音素の実登録数を記録した。
test通常文EERは1.769→1.361→1.497%、別テキストは1.531→1.221→1.020%。
両testで本人スコアと本人−他人の平均差が増加したが、通常文は20→30でEERが少し上がった。
30−10の差の95% CIは両方とも0を含み、改善の確証には至っていない。
[比較条件と再現手順](phoneme-all-enrollment-scaling/README.md)、[HTML比較表](phoneme-all-enrollment-scaling/evaluation-results.html)、
[測定表](phoneme-all-enrollment-scaling/evaluation-results.md)、[結果の解釈](phoneme-all-enrollment-scaling/interpretation.md)を参照する。

Issue #22の追加検証では、全36音素encoder・登録上限30区間/音素を固定して、等重み・音素別MLP・音素間Transformerを比較する。
JVS＋Common Voiceの学習140話者だけで各方式3 seedを学習し、validationで選んだモデル・閾値をtestへ固定適用する。
Transformerのtest平均EERは通常1.497→1.186%、別文1.020→0.941%。3 seedの最良値は通常0.952%、別文0.747%で、それぞれ別のseedである。
一方、部分ノイズでは3.356%・3.401%へ悪化した。MLPはvalidationで初期の等重みが最良だった。
[設計と再現手順](phoneme-transformer-fusion/README.md)、[HTML比較表](phoneme-transformer-fusion/summary.html)、
[全seedの測定結果](phoneme-transformer-fusion/evaluation-results.md)、[結果の解釈](phoneme-transformer-fusion/interpretation.md)を参照する。

同じencoder・3 seed・学習ペア列・4,000更新・clean validation採用規則を固定し、trainにノイズと子音欠損を追加して再測定した。
Transformerのtest平均EERは、ノイズ時に通常3.356→1.374%、別文3.401→1.166%へ低下した。
一方、cleanは通常1.186→1.429%、別文0.941→1.136%、子音欠損は通常1.523→1.769%、別文0.887→1.166%へ上昇した。
3 seedともノイズ時は改善したが、等重みのノイズEER（通常1.302%、別文1.093%）には平均で届いていない。
ノイズへの弱さを学習で緩和できた一方、clean性能を保つという目的は達成できず、全条件での上位互換としては採用しない。
[追加学習の固定条件](phoneme-fusion-augmentation/README.md)、[HTML比較表](phoneme-fusion-augmentation/evaluation-results.html)、
[全seed・固定閾値FAR/FRR・信頼区間](phoneme-fusion-augmentation/evaluation-results.md)を参照する。

総括後の追加検証では、全36音素encoder・登録上限30区間を固定して、5母音必須・4母音以上・3母音以上かつ母音と子音を合わせて5種類以上を比較した。
等重みと既存clean学習Transformerの3 seed、計12条件のvalidation閾値を固定してからtestを測定した。
主比較の等重み・4母音以上では、通常文の全入力FRRが4.400→2.400%、別文が14.222→2.667%へ低下した。
全入力FARは通常文0.410→0.419%、別文0.714→0.810%で、他人受入はそれぞれ1件・6件増加した。
別文の追加他人受入6件のうち5件は /u/ 欠損入力だった。3母音以上＋合計5種類は別文FRR 1.556%だが、4母音条件から追加した入力は4話者・5発話に限られる。
新規学習なし・観測済みJVS testの探索で、運用への採用やFAR維持の確証ではない。
[固定条件](phoneme-all-missing-vowel-evaluation/README.md)、[HTML比較表](phoneme-all-missing-vowel-evaluation/evaluation-results.html)、[結果の解釈](phoneme-all-missing-vowel-evaluation/interpretation.md)を参照する。

母音不足対応については、さらに既存学習70 client_idを除くCommon Voice 60話者ラベルを新規に固定し、校正30話者・test30話者で評価した。
各話者20発話で登録、30発話で照合し、全36音素encoder・等重み・登録上限30区間を維持した。
5母音必須→4母音以上でtest全入力FRRは44.222→23.444%、FARは0.441→0.640%。差の95%区間はそれぞれ[−24.889, −17.000] pp、[+0.071, +0.402] ppだった。
4母音FARの95%区間上端1.524%とFAR増加上端0.402 ppが事前の研究用基準を超え、現構成への単純導入は見送ると結論づけた。
JVS閾値の移植時には4母音FARが7.165%となり、入力不足の救済だけで収録集団への一般化と校正の問題は解消しなかった。
[事前固定した設計](phoneme-missing-vowel-holdout/README.md)、[測定表](phoneme-missing-vowel-holdout/evaluation-results.md)、[結論](phoneme-missing-vowel-holdout/interpretation.md)を参照する。

## 6. 基本方針

- 音素方式が既存方式より優れているとは仮定しない。
- 各フェーズを再現可能な形で実施する。
- 学習用データと評価用データの混入を防ぐ。
- 新規ユーザーの追加時に共通モデルを再学習しない。
- 類似度と本人である確率を区別する。
- 各フェーズの品質を確認してから次へ進む。
- 新たな学習データは[採用手順](../docs/training-data-adoption.md)に従い、train候補の人間レビューと採用判断を完了してから使う。

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
