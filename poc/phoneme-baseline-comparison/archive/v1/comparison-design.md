# Phase 7 比較条件の設計

設計日: 2026-10-06。対象は[Issue #45](https://github.com/muzin00/phonia/issues/45)。
[比較protocol](config/comparison-protocol.json)を条件の正本とし、Phase 6の固定入力・モデル・評価処理を引き継ぐ。
本工程で実施したのはvalidationの入力metadata確認までである。
比較用profile・認証score・閾値は未作成であり、実装・validation検証後の実行計画凍結を別途必要とする。

その後の[短入力確認](../../short-input-results.md)で、`ecapa_time_exact`の30 ms登録入力は推論エラーとなった。
本書とprotocol 1.0.0は初回設計を保持する。時間合わせ補助条件は再設計・新protocol版の検証を必要とし、
現在の条件のままtest推論・実行計画凍結へは進めない。

## 1. 比較する方式と確認する問い

主比較は、同じ登録元WAV集合・同じ照合WAVを与えた次の3方式とする。

| ID | 入力 | encoder | 扱い |
| --- | --- | --- | --- |
| `vowel_exact` | Phase 6と同じ境界内の母音区間 | 採用済みstatistics MLP | 既存方式の基準 |
| `vowel_context20` | 同じ母音区間に前後各20 msを追加 | 同じ固定statistics MLP | 文脈追加の診断 |
| `ecapa_whole` | 各元WAVの発話全体 | 選定済みSpeechBrain ECAPA-TDNN | 既存の発話全体方式 |

この比較が確認するのは「固定した各システムを同じ取得音声・trialで使用した場合の認証性能」である。
発話全体方式は母音方式より長い音声を利用するため、利用時間を揃えた次の2方式も補助比較として測定する。

| ID | 入力 | 対になる方式 |
| --- | --- | --- |
| `ecapa_time_exact` | 元WAVごとに境界内母音方式の実利用時間と同じ長さの中央区間 | `vowel_exact` |
| `ecapa_time_context20` | 元WAVごとに文脈母音方式の実利用時間と同じ長さの中央区間 | `vowel_context20` |

この2方式は短縮入力のECAPAであり、発話全体方式の主結果と区別する。
5方式すべてを事前に定義し、testの結果から方式・文脈幅・登録数を選び直さない。
文脈幅の探索、encoderの追加学習、小型ECAPAとのサイズ比較は今回の条件に含めない。

## 2. モデルと前処理

母音方式はPhase 6の`statistics_mlp`、seed `20260926`、JVS train 70話者の採用bundleを使用する。
128次元、65,920 parameters。checkpointと学習済み特徴統計のSHA-256をprotocolへ保持する。
24 kHz、64-bin log-mel、既存のDC除去・特徴統計正規化・最大250 msのcenter cropを維持する。
RMS正規化、モデル・特徴統計の更新は行わない。

ECAPAは`speechbrain/spkrec-ecapa-voxceleb`のrevision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`を使用する。
192次元、embedding encoderは20,767,552 parameters。
重み、PCM16からfloat32への変換、24→16 kHzのsinc resamplingは
[動作確認設定](../../config/embedding-smoke.json)へ固定済みである。
`encode_batch(..., normalize=False)`でembeddingを抽出し、後段でL2正規化する。
保存されたembedding統計による正規化、VAD、無音除去、RMS正規化、PLDA、AS-Normを追加しない。

モデル規模は約315倍、学習データはJVSとVoxCeleb、入力帯域は24 kHzと16 kHzで異なる。
アーキテクチャ、損失、時間方向の集約も異なるため、性能差を音素分割だけの効果とは解釈しない。
パラメータ数から結果を予測せず、これらの差を結果表に併記する。
VoxCelebとJVSの話者の個人単位での重複は未確認である。

## 3. 文脈区間の定義

登録と照合の両方で、Phase 6の利用可能な元区間を同じanchorとして使う。
新しい母音検出、隣接音素の再ラベル、欠落母音の補充は行わない。
元区間の品質判定を維持し、文脈追加後のRMSを理由に区間を選別し直さない。

24 kHzの元WAVに対して、anchorの半開区間を`[s, e)`、総frame数を`F`とすると、
文脈区間は`[max(0, s-480), min(F, e+480))`。
拡張後の長さが6,000 framesを超える場合は、既存のcenter cropと同じ
`floor((length-6000)/2)`のoffsetで6,000 framesを取る。その後に既存の特徴抽出を適用する。
境界内方式にも既存の6,000-frame cropを適用して実利用時間を算出する。

長いanchorではcropにより文脈が一部または全部残らない。
元境界、拡張境界、実効境界、crop有無、元境界の外側に残ったframe数を記録する。
本条件は母音で学習した固定encoderに文脈を与える診断であり、文脈入力で再学習した方式の評価ではない。
既存Phase 4/5の入力検証を弱めず、Phase 7の専用adapterで元anchorの整合性を確認して拡張波形を作る。

## 4. 分割・登録・照合入力

train 70、validation 15、test 15話者の既存splitを維持する。新しい学習は行わない。

| role | 発話集合 | 使用目的 |
| --- | --- | --- |
| enrollment | parallel100の001〜050 | 登録profile |
| verification | parallel100の051〜100 | 通常文評価、validationのみ閾値較正 |
| cross_text_verification | nonpara30の全発話 | 異文評価、較正には使用しない |

登録条件は各母音1・5・10区間、主条件は10区間。
Phase 6のseed `20260930`で決めたrank列の先頭1・5・10を再利用し、同じ区間の入れ子を維持する。
区間長、品質score、認証scoreによる再選別は行わない。

ECAPAの登録元WAV集合は、その登録条件のanchorが含まれる元WAVの集合とする。
同じWAVに複数のanchorがあっても発話embeddingは一度だけ使う。
したがって「登録1」はECAPAの1発話を意味せず、各母音1区間を提供した元WAV集合を意味する。
元WAV数・取得時間・実利用時間を登録条件ごとに保存する。
validationでは10区間条件の元WAV数は29〜37、利用時間の中央値は母音3.52秒／発話全体238.12秒だった。
詳しい入力量は[validation入力確認](../../validation-readiness.md)を参照する。

照合は1元WAVを1queryとし、その発話の利用可能な全anchorを使う。
複数発話の結合、母音数の人工的な同数化、区間の間引きを行わない。
ECAPAの発話全体方式は同じqueryの元WAV全体を使う。
登録と照合の同一WAV・同内容コピーの混入を元音声checksumで検査する。
登録に必要な全話者・全母音の区間が不足した場合は停止し、話者を削除して続行しない。

## 5. 利用時間を揃える補助条件

方式`m`の実効区間を元WAV `f`ごとにまとめ、区間の**和集合**のframe数を`B(f,m)`とする。
文脈区間の重複を一度だけ数え、cropで捨てたframesを含めない。
これを物理的な利用音声量とし、重複を含むencoder処理時間の合計も別の値として記録する。

ECAPAには同じ元WAVの`[floor((F-B)/2), floor((F-B)/2)+B)`を渡す。
登録は各元WAV別、照合はその1元WAVについて時間を合わせ、24 kHzで切ってから16 kHzへ変換する。
resampling後の実frame数と時間も記録する。
波形の連結、反復、人工的なpadding、音声の品質を見ての切り出し位置変更は行わない。
`B=0`は`no_score`。正の入力で推論例外・非有限embedding・零ノルムが出た場合は実装エラーとして停止する。

この条件は元音声と利用時間を揃えるが、音素内容、入力帯域、区間配置、集約方法は揃わない。
中央区間が無音を多く含む場合も選び直さず、その制約を報告する。
validationの登録中央区間は境界内時間合わせで最短30 ms、文脈時間合わせで最短70 msとなる。
短入力pilotでは30 msがpaddingエラーとなり、選定した70 ms入力は成功した。
時間合わせ補助条件を再設計し、test前に条件の変更理由と新しいprotocol版を記録する。

## 6. 登録profileとscore

各embeddingのL2正規化を`u(x)=x/||x||`とする。
母音方式のprofileは、母音ごとに区間embeddingの等重み平均を再びL2正規化する既存式を使う。
queryは母音別に各区間とprofileのcosineを平均し、5母音のscoreを等重み平均する。
queryの平均embeddingを再正規化する別の式へ変更しない。

ECAPAのprofileは`u(mean_f(u(embedding(f))))`。元WAV単位の等重み平均とし、音声長で重み付けしない。
queryの正規化embeddingとprofileの内積をscoreとする。
時間合わせの2方式もそれぞれの中央区間から独立したprofileを作り、同じ集約式を使う。
計算順は母音区間ID順／ECAPA元WAV辞書順、平均・L2正規化・scoreはfloat64とする。
零ノルムや不正数値は停止条件である。

## 7. 共通trial・入力不足・分母

Phase 6のquery ID、trial ID、正解ラベル、登録数を再利用する。
各queryを同じsplitの全15話者と比較し、genuine 1件、impostor 14件を作る。
方式別score IDはcanonical JSONの`SHA256([protocol_version, method_id, phase6_trial_id])`。
名目件数は1 splitあたり1,200 queries、3登録条件で54,000 trials。
5方式で270,000 score記録、validationとtestを合わせて540,000記録となる。

母音2方式は5母音のいずれかがないqueryを`no_score`とする。文脈で不足を補わない。
発話全体ECAPAは母音の存在を受入条件にしない。時間合わせECAPAは予算が正ならその区間を使う。
全方式で全候補queryを保持し、`no_score`は認証として受け入れない。
不正WAV、参照範囲、checksum、split、embeddingの不一致は停止し、精度計算から黙って除外しない。

| 指標 | 分母と計算 |
| --- | --- |
| query coverage | scoreありquery / 全query |
| 条件付きFAR | scoreありimpostorの誤受入れ / scoreありimpostor |
| 条件付きFRR | scoreありgenuineの誤拒否 / scoreありgenuine |
| 全入力FAR | 誤受入れ / 全impostor |
| 全入力FRR | (誤拒否 + no_score genuine) / 全genuine |

主結果は各方式が元来処理できる入力でscoreを作り、全入力の分母で評価する。
補助診断では、5方式すべての入力条件を満たすqueryの共通集合でも比較する。
この集合はscore値から決めず、同じ規則を各split・roleへ適用する。
共通集合用の閾値はvalidation / verificationの共通集合で別途較正し、testと異文へ固定適用する。
各話者・roleでscoreありqueryが0件の条件は「評価不能」とし、話者を除かない。

## 8. 閾値・評価表・信頼区間

方式・登録数・入力集合ごとに、validation / verificationのscoreありtrialから閾値を決める。
目標FAR 1%を主動作点、0.1%とEER operating thresholdを補助動作点とする。
Phase 3の`far_target_threshold`・`eer_operating_threshold`を再利用し、`score >= threshold`で受け入れる。
同点は既存のfloat64 `nextafter`規則でまとめて扱う。
test / verificationとvalidation・testのcross-textへ同じ閾値を適用し、再較正しない。
testの実測FARが目標を超えた場合もそのまま報告する。

**主指標は登録10、test / verification、validation目標FAR 1%での全入力FRR**。
方式間で入力不足率が異なるため、Phase 6の主指標だった条件付きFRRから全入力FRRへ変更する。
条件付きFRR、実測条件付きFAR、全入力FAR、coverage、件数を同じ閾値で必ず併記する。
EER・ROC・DETはscoreありtrialのみで算出し、coverageを並べる。
新たな精度の合否基準は設けず、悪い結果も既定の全条件で報告する。

5方式 × 3登録数 × 2role × 2splitの60条件を全入力と共通入力のそれぞれで報告する。
元query長の区分は`[0,3), [3,5), [5,10), [10,∞)`秒、
queryの実利用時間は`[0,0.5), [0.5,1), [1,2), [2,5), [5,∞)`秒。
時間合わせの対比較は基準の母音方式のquery和集合時間で両者を同じ区分へ割り当てる。
区分内では閾値を再較正せず、空・scoreなしの区分も「評価不能」として残す。
母音別scoreは母音方式の診断値として記録し、発話全体方式と母音別EERを直接比較しない。

test 15話者を単位に10,000回、PCG64・seed `20260929`でspeaker bootstrapを行う。
話者IDは辞書順、genuine重みはquery話者の出現数`n_i`、impostorは`n_i*n_j`。
異なる話者が2人未満のdrawは引き直し、全方式・条件で同じdrawを使う。
閾値は固定し、95% percentile区間をNumPy `method="linear"`で算出する。
指標の区間に加え、各方式−`vowel_exact`と、時間合わせECAPA−対応母音方式のFRR差をpercentage pointsで報告する。
この区間は学習・較正の不確かさを含まず、低FARでは試行分解能と誤受入れ件数も併記する。

## 9. 実装、凍結、実行の順序

母音推論は既存Phase 6環境のtorch 2.14.1、ECAPA推論はPhase 7環境のtorch/torchaudio 2.11.0で行う。
両者ともCPU float32、batch 1、intra/inter-op threads 1、deterministic algorithmsを使用する。
score・指標の共有処理と各環境lockを固定し、既存Phase 6成果物を変更しない。
既存の境界内方式のprofile・score・全入力用閾値は、入力・モデル・実装・環境・IDのhashが一致する場合に再利用する。
共通入力用の閾値とPhase 7の条件表・信頼区間は今回の規則で作成する。

1. prepare: protocol、元音声、anchor品質、登録／照合分離を確認し、実効区間と元WAV別予算を保存する。
2. validation pilot: 辞書順先頭3話者とmetadataで選んだ最短・最長入力を使い、5方式の推論・profile・scoreの成立性を確認する。
   登録数1・5・10、通常文・異文を含め、短い中央区間と長区間cropを検証する。
3. validation: 全15話者でprofile・query・score・閾値を作成し、全入力と共通入力の結果を保存する。
4. freeze: 実装・protocol・依存lock・モデル・統計・全元音声・派生区間・validation成果物、
   testの登録/query/trial metadata、再利用成果物、時間・メモリ・cache計画をhash付き実行計画へ保存する。
   新しい方式によるtest音声の推論前に凍結する。
5. test: 凍結後の新方式を一度の評価実行として計算し、固定閾値の結果と再利用した境界内結果を集計する。
6. report: 全条件、信頼区間、曲線、入力量、失敗理由、実行時間、モデル・profileの不変を検証して記録する。

adapterとcacheの検証には、既存APIとの境界内score一致、独立計算との集約式一致（絶対誤差`1e-12`）、
登録数の入れ子、区間clip/crop、重複の和集合、時間一致、同点閾値、no_score分母、固定閾値適用を含める。
推論の反復再現性とモデル・特徴統計不変をpilotで確認する。
cache keyにはモデル・前処理・元音声・実効frame範囲のhashを含める。
時間はmodel load、特徴抽出を含むembedding推論、profile/score、全工程を分け、warmup後の計測範囲を明記する。
音声取得時間と演算時間を別の指標として報告する。

## 10. 保存する成果物と解釈の限界

保存先は`artifacts/phoneme-baseline-comparison/<run-id>/`。未使用のrun IDを使い、上書きしない。
実行計画、モデルmanifest、区間・時間予算、query/trial一覧、方式別profile・embedding・score、
validation閾値、指標・曲線・bootstrap、環境・時間・不変確認を保存する。
scoreからquery、元WAV・区間、profile、モデル、閾値へ追跡できるIDとchecksumを持たせる。
再利用したPhase 6ファイルは保存先とhashを記録し、複製する場合も同一性を確認する。
失敗runを残し、コード・条件変更は別runとする。test観測後の修正再実行は独立確認と報告しない。

JVSの統制収録内での別発話評価であり、収録日・端末・個別sessionの情報はない。
異文条件をcross-sessionと呼ばない。Phase 3・6でtestを観測済みであり、新しい未観測holdoutとは扱わない。
今回の設計で新たなtest scoreは読まず、入力の成立性はvalidation metadataから確認した。
音素分割の因果効果やモデルサイズのスケーリングを確認するには、学習条件を揃えた別の対照が必要である。
