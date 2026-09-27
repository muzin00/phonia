# 学習・評価設計

## 1. 目的

入力方式やencoderを公平に比較し、学習に含まれない話者の本人照合性能から方式を選択する。
validationをモデル選択に使用し、testは全設定を固定した後の最終評価だけに使用する。

## 2. 候補を比較する学習条件

各候補設定では入力、encoder、RMS正規化、損失の候補軸を組み合わせる。samplingは固定する。候補軸以外の
差を抑えるため、比較時は次を共通化する。

- train cohortと使用する区間ID
- optimizer、学習率schedule、最大更新回数と停止規則（実更新数は停止時点で異なり得る）
- checkpoint評価の間隔とearly stopping規則
- validationの登録区間、照合区間、trial一覧
- seedの集合

候補軸、組み合わせ、採用規則は[候補比較計画](comparison-plan.md)を正本とする。
機械可読な共通手順は`config/experiment-protocol.json`とする。候補構造は各encoder JSON、
学習参照値は`config/baseline-log-mel.json`を参照し、展開したrun設定に矛盾があれば実行を止める。

まず10話者cohortで損失が低下し、登録・照合処理まで実行できることを確認する。入力方式の
選択は70話者cohortで行う。互換性のある候補の組み合わせを3 seedで学習し、単一の初期値や
単一項目だけの比較で判断しない。

front-endの違いにより完全に同一条件にできない場合は、差と理由を実験結果へ記録する。

## 3. encoderと学習目的

### 3.1 参照encoder

最初の学習成立性確認に使う参照候補として、5母音で共有する小型TDNNを定義する。最終的な
encoderは、統計pooling + MLP、TDNN、framewise CNN、および生波形用encoderを同じ選択規則で比較して決める。

| 段 | 仕様 |
| --- | --- |
| 入力 | 64次元log-Mel |
| TDNN 1 | Conv1d 64→128、kernel 3、dilation 1 |
| TDNN 2 | Conv1d 128→192、kernel 3、dilation 2 |
| TDNN 3 | Conv1d 192→256、kernel 3、dilation 3 |
| 各block | same padding、channel方向LayerNorm、GELU、dropout 0.1 |
| pooling | mask付き平均と標準偏差の連結（512次元） |
| projection | Linear 512→256、GELU、dropout 0.1、Linear 256→128 |
| 出力 | 128次元をL2正規化したspeaker embedding |

log-Mel Conv1dは`bias=false, stride=1, groups=1`とする。

最初の畳み込み前と各TDNN blockの直後に無効時刻を0へ戻す。LayerNormは各時刻のchannel方向だけに適用し、
時間方向やbatch方向の統計へpaddingを混ぜない。最短の2 frameでもpoolingが成立するよう、
標準偏差は`sqrt(max(population_variance, 1e-5))`で計算する。poolingの統計は有効時刻だけを
対象とし、Linearはbiasあり、L2正規化は`x / max(norm(x), 1e-12)`とする。

母音ラベルはencoderへの入力にせず、samplingとcontrastive lossの組を作るためだけに使う。
母音embeddingによる追加条件付けは予算上今回は探索しない。不要性を確認したわけではない。
母音別profileはencoderの外で管理する。AAM headは母音によらず共通の話者分類である。

最小の性能・コスト基準として、各Mel binのmask付き平均と標準偏差を連結した
128次元を`Linear 128→256 -> GELU -> dropout 0.1 -> Linear 256→128 -> L2 normalization`
へ入力する`statistics_mlp`も比較する。65,920 parameterのMLPと411,136 parameterのTDNNの差を
時間構造だけの効果と解釈しない。

限定比較では`framewise_cnn`を追加する。kernel=1、stride=1、padding=0のConv1dを
64→256→448→256と積み、TDNNと同じchannel LayerNorm・GELU・dropout・pooling・projectionを
使う。411,904 parameterでTDNNとの差は約0.19%。pooling前に時間を混ぜず非線形特徴変換を行う。
幅の差は残るが、MLPよりモデル規模とpooling位置を揃えた対照となる。log-Mel 3候補の構造は
`config/log-mel-encoders.json`を正本とする。

### 3.2 生波形encoder候補

生波形は4層のvalid Conv1dで時間方向をdownsamplingし、log-Mel TDNNと同じmask付き
mean / std pooling、256次元hidden、128次元embeddingを使う。

第1層の適切な時間幅は事前に一意に決めにくいため、次の2候補を比較する。

- `waveform_cnn_k80`: 80 sample（3.33 ms）kernel、最終受容野9.83 ms
- `waveform_cnn_k240`: 240 sample（10.00 ms）kernel、最終受容野16.50 ms

両候補は最短30 ms入力で最終時系列をそれぞれ8 frame、6 frame残す。第1層以外のchannel、
kernel、stride、normalization、projectionは共通化する。構造、出力長、maskの伝播規則は
[生波形encoder設計](waveform-encoder-design.md)と
`config/waveform-encoders.json`を正本とする。

どちらの受容野が話者特徴に有利かは固定せず、RMS正規化とlossの候補を組み合わせた
validation macro EERで決める。

限定比較の`waveform_cnn_k240_context27`はk240へdepthwise context層を追加し、公称受容野
85.83 msを扱う。RMSあり・AAMのみで学習し、同条件のk240を対照とする。詳細は生波形設計に従う。

### 3.3 参照batch sampling

1 batchを`P=10`話者、5母音、各話者・母音につき`K=2`区間の100区間で構成する。

- 話者は既選出回数が少ない順に10人を重複なしで選び、同回数はseed付きhash順で選ぶ。
- 各話者について5母音を必ず含める。
- 話者・母音内は区間IDをseed付きhash順に並べて先頭2件を抽出し、batch内で重複させない。
- 話者の累積選出回数差を1以内に保つ。25話者cohortの端数でも話者を捨てない。
- このversionでは区間長bucketingを使用しない。母音内はspeaker ID、segment IDの順に配置する。
- trainの`evaluation_role=training`以外はsamplerへ渡さない。

これにより各anchorは、同じ話者・同じ母音のpositiveを1件、同じ母音・異なる話者の
negativeを18件持つ。区間数の多い話者や母音が更新回数を支配しない。

samplingは`P=10, K=2`に固定する。`P=5, K=4`より同母音の異話者negativeを多く確保でき、
各anchorに必要なpositiveも1件存在するためである。

hashは`SHA-256(UTF-8の空白なしJSON配列)`を用い、Pythonでは
`json.dumps(values, ensure_ascii=False, separators=(',', ':'))`相当とする。
話者の同回数tieは`["speaker", run_seed, logical_update, speaker_id]`、区間順は
`["segment", run_seed, logical_update, speaker_id, vowel, segment_id]`で決める。
hash同値はIDのUTF-8辞書順とする。logical updateは0始まりで、モデルごとに乱数消費が違っても
同じseed・cohort・updateの抽出IDとcropを一致させる。同一発話由来のpositiveを禁止する設計では
ないため、その割合も記録し、別発話・別セッションへの一般化と混同しない。

### 3.4 損失関数

参照候補の学習目的は次の和とする。

```text
L = L_AAM-Softmax + 0.5 * L_SupCon-within-vowel
```

- `L_AAM-Softmax`: train話者をclassとする。正規化済みembeddingとclass weightを使い、
  scale `30`、angular margin `0.2`とする。
- `L_SupCon-within-vowel`: temperature `0.07`とし、同じ母音の中だけでpositiveとnegativeを
  作る。positiveは同一話者、negativeは異なる話者とする。
- AAM-Softmaxのclassification headは学習時だけ使用し、登録・照合成果物へは含めない。

`AAM-Softmaxのみ`と`AAM-Softmax + 母音内SupCon`を候補とする。SupConのみは除外し、
話者分類目的を共通にした上で指定したmetric lossを追加する価値を比較する。重み0.5と
temperature 0.07は予算上固定した参照値であり、SupCon全般の有効性を結論しない。

anchor iの同母音・同話者・別区間集合をP(i)、同母音でi以外の全区間をA(i)とする。
`l_i = -mean_{p in P(i)} log(exp(z_i·z_p/tau) / sum_{a in A(i)} exp(z_i·z_a/tau))`を
logsumexpで計算し、全anchorの平均をSupConとする。分母にはpositiveも含め、自己比較は含めない。
temperatureによる追加の倍率は掛けない。positiveが空なら失敗とする。
AAMは全区間のcross entropy平均とし、標的logitを`30*cos(theta+0.2)`へ置き換える。
cosineはacos前に`[-1+1e-7, 1-1e-7]`へclampし、easy-margin等の別式は使用しない。

### 3.5 メモリ制約と同値なバッチ分割

過学習専用runを除く全runで論理batch100区間を作り、母音順a/i/u/e/oの20区間ずつ処理する。
各母音内の10話者×2区間を分割せず、`(L_AAM_v + lambda * L_SupCon_v) / 5`をbackwardする。
5回のbackward後に一度だけgradient clip・optimizer step・scheduler step・zero_gradを行う。
AAMのみの場合も同じ分割を使う。論理update数はoptimizer step数で数える。
channel LayerNorm等にbatch依存統計がなく、SupConも母音間の組を使わないため、これは同じ目的関数を
分解する方式である。dropoutの乱数配置や浮動小数点加算順まで100区間一括と同一とは主張しない。

任意のmicrobatch分割や、embeddingをdetachして集約する代替は禁止する。20区間でも資源上限を
満たさない場合は今回の実行を保留し、全候補共通の別precision等を次の事前登録版で検証する。
現versionはfloat32とし、特定候補だけprecisionを変更しない。
dropout無効の固定入力で、一括と母音別分割のloss・parameter gradientが
`atol=1e-6, rtol=1e-4`内で一致することを実装テストにする。

### 3.6 共通するoptimizerと更新条件

| 項目 | 成立性確認 | 本学習 |
| --- | ---: | ---: |
| cohort | 10話者 | 70話者（学習曲線は10 / 25 / 50も同条件） |
| 最大更新数 | 2,000 | 30,000 |
| optimizer | AdamW | AdamW |
| 学習率 | `3e-4` | `3e-4` |
| weight decay | `1e-4` | `1e-4` |
| warmup | 100 update | 1,000 update |
| schedule | warmup後cosine、下限`1e-5` | warmup後cosine、下限`1e-5` |
| gradient clip | global norm `5.0` | global norm `5.0` |
| validation間隔 | 250 update | 1,000 update |

early stoppingはvalidation macro EERが8回連続で改善しない場合とする。改善は絶対値で
`0.001`以上の低下と定義する。最大更新数へ達した場合も終了する。checkpoint選択には
smoothed値ではなく各評価時点の実測macro EERを使う。
early stopping用の基準値は初回評価値で初期化し、そこから0.001以上低下したときだけ基準値と
patienceを更新する。これは最小EERのcheckpoint保存とは別状態で管理する。
scheduleは事前の最大更新数を基準とし、early stopping後に再伸縮しない。
AdamWはbetas=(0.9, 0.999)、eps=1e-8、全trainable parameterへ同じweight decayを適用する。

seedは`20260926`、`20260927`、`20260928`の3個を本比較に使う。データ順、crop、モデル初期化、
workerのseedをrun seedから決定論的に導出する。cropはworkerの処理順に依存させない。
再現性を優先する比較runでは、利用する
frameworkのdeterministic modeを有効にし、非決定的な演算を検出したら失敗させる。
Python / NumPy / framework CPU・deviceをrun seedで初期化する。worker初期化seedは
`["worker", run_seed, worker_id]`のhash整数の下位32 bitとし、入力の選択・cropには使用しない。
成立性確認はseed `20260926`だけを使う。

## 4. 学習後の登録・照合評価

### 4.1 encoderの固定

評価するcheckpointを読み込み、encoderと入力正規化を固定する。validationの音声を使った
encoderの更新や正規化統計の再計算は行わない。

### 4.2 話者プロファイルの作成

validationの`enrollment`から、話者別・母音別に登録区間を取得する。初期の主条件は
1話者・1母音あたり10区間とする。区間IDと抽出seedを固定し、すべてのモデルで同じ区間を
使用する。

抽出seedは`20260930`。話者・母音ごとにsource_file単位で区間をまとめ、source順を
`["enroll_source", seed, split, speaker_id, vowel, source_file]`、source内区間順を
`["enroll_segment", seed, split, speaker_id, vowel, source_file, segment_id]`のhashで固定する。
source順を巡回して各sourceから未選択区間を1件ずつ取り、10件まで繰り返す。
10発話未満でも使用可能な全sourceを一巡するまでは同じsourceから2件取らない。
区間が10件未満なら評価前に停止する。品質属性・区間長・モデルscoreで選別しない。

各区間から得たembeddingをL2正規化し、その平均を再度L2正規化して話者・母音別の
登録プロファイルとする。登録時にencoderは再学習しない。

登録区間数による変化は、上記固定順の先頭1、5、10区間を使う入れ子条件として評価する。
profileの正規化もeps=1e-12とする。平均embeddingのnormがeps未満なら評価を停止する。

### 4.3 trialの作成

`verification`の各照合区間について、同じ母音の15話者分の登録プロファイルと比較する。
1区間につき、同一話者との1試行をgenuine、異なる14話者との14試行をimpostorとする。

登録と照合で同じ元音声を使用しない。trial一覧を事前に成果物として固定し、入力方式間で
共通利用する。

主要評価とは別に、`cross_text_verification`を同じ方法で照合し、登録文と異なる文章での
頑健性を確認する。JVSには収録日とセッションIDがないため、この結果をcross-session性能
とはみなさない。

### 4.4 スコア

初期スコアは、照合embeddingと同じ母音の登録プロファイル間のcosine類似度とする。
母音ごとのスコアを一つの認証スコアへ統合する方式は
[Issue #22](https://github.com/muzin00/phonia/issues/22)で扱う。

### 4.5 指標

validationで入力方式とcheckpointを比較する主指標は、5母音の母音別EERを単純平均した
macro EERとする。区間数の多い母音だけが選択結果を支配しないようにする。

併せて次を記録する。

- 母音別EER
- pooled EER
- ROC曲線とDET曲線
- FARとFRR
- validationで選んだ固定閾値におけるFARとFRR
- `verification`と`cross_text_verification`の差
- 照合区間長別の性能: 30–49 ms、50–99 ms、100 ms以上
- 登録区間数別の性能: 1、5、10区間
- 学習seedごとの値、平均、ばらつき
- 学習時間、推論時間、最大メモリ、パラメータ数
- 照合区間の`is_devoiced`、`is_long`、各`quality_flags`の有無別の性能・件数・話者数
- 250 ms超でcropされた区間と、それ以外の性能

FARとFRRは別々に報告する。genuineとimpostorの試行数が異なるため、両者の誤り件数を
単純に合算したaccuracyを主指標にしない。

属性はmanifestの機械ラベルであり、`is_devoiced=false`を有声と確認した正解とはみなさない。
欠損値はunknownとし、フラグ別集計は重複を許す。母音ごとのgenuine / impostor数、query数、
話者数を併記し、片方のtrial種別が空ならEERはNAとする。5母音のいずれかがNAならその層の
macro EERもNAとし、欠けた母音だけを除外して通常のmacro EERと比較しない。
収録条件属性がなければunknownとして限界を記載する。評価から学習区間の採否を変更しない。

### 4.6 指標の数値定義と不確実性

encoderの出力をCPUのfloat64へ変換してprofile・cosine scoreと集計を計算し、
`score >= threshold`を受理とする。母音別に全trialを同じ重みで
集計し、話者macro平均ではないことを明記する。pooled値は5母音のtrialを連結して算出する。
同値scoreは一括して扱い、全受理・全拒否を含むROC上で`FAR - FRR = 0`を挟む隣接点を
線形補間してEERを求める。完全一致点があればその値を使う。EERの補間値と実際に使える
離散閾値は区別する。指標テストには完全分離・全score同値（EER=0.5）・scoreのtieを含める。

各候補と平均EER最小候補の差には、同一trialを使う対応付き話者bootstrapを必ず併記する。
seed `20260929`で15話者を復元抽出し、10,000 replicateを生成する。同一の抽出を全候補・
学習seedへ使う。話者sの抽出回数をn_sとし、genuine trialの重みはn_s、impostor s→tは
n_s*n_tとする。元々異なる話者のtrialだけを再重み付けし、複製IDを別人にしない。
2人未満しか異なる話者を含まない抽出は再抽出する。重み付き母音別EER→5母音平均→3 seed平均
の順で差を求め、2.5 / 97.5 percentile（linear補間）の区間を報告する。seed自体は再抽出しない。
speaker ID辞書順のリストからPCG64で抽出し、ライブラリversionと抽出された話者ID列も保存する。
区間は選択後の記述的・pointwise推定であり、18候補の多重比較を補正した有意差や同等性の検定ではない。
登録区間集合を再抽出しない条件付きの不確実性推定であり、3 seedや多数trialを独立話者数と
みなさない。区間が重なることを統計的同等性の根拠にしない。採用規則は候補比較計画に従う。

### 4.7 再学習しない感度診断

採用設定の70話者3 checkpointに対しvalidationのみで実施する。登録profileは元のまま固定し、
`verification`のqueryだけを変更する。元trial IDに変換IDを付けた派生trialを保存する。

- 境界: 元WAV上のstart / endを同時に±120 / ±240 sample（±5 / ±10 ms）移動する。
  長さは変えず、通常のcrop・前処理を後から適用する。WAV範囲外ならclampせず対象外とし、
  全4変換で範囲内の共通集合に対する無変換scoreを対照にする。除外数を話者・母音別に報告する。
  これは平行移動感度だけであり、開始・終了境界の独立誤差すべての検証ではない。
- gain: crop後・DC除去前の浮動小数点波形を`10^(±6/20)`倍する。clipせず、その後は
  選択済みのDC・RMS・特徴正規化を使う。RMSやLayerNormで差が抑えられる場合も含めて測る。

各条件の母音別・macro EERと元条件との差、固定validation閾値のFAR / FRRを報告する。
閾値や特徴統計を変換条件に合わせて再推定しない。元manifestは書き換えず、手作業で境界を
正解へ修正する処理や学習augmentationと混同しない。診断結果による候補再選択は行わない。

### 4.8 計算量benchmark

validationの主verification queryを`["benchmark", 20260930, segment_id]`のhash順に並べ、
先頭1,000件（不足時は全件）を全候補共通で使う。WAV読出しを除き、入力前処理とencoderを含める。
同じdevice・float32・batch=1で、1周warmup後5周計測し、deviceを同期して1区間当たり時間の
中央値をrun値とする。profile作成は含めない。専用プロセスでencoder読込後にpeak counterを
resetし、同じ計測中の最大メモリを記録する。backendの計測API・device・ソフトウェアversionを
`execution-budget.json`に事前固定し、比較途中で変更しない。

## 5. checkpointと閾値の選択

各学習runでは、決められた間隔でvalidationの登録・照合評価を行う。主条件のmacro EERが
最小のcheckpointを選ぶ。完全同値ならupdateが最も早いものを選び、副指標で選び直さない。

実運用を想定した判定閾値はvalidationだけで選択する。初期PoCでは、EER閾値に加え、
目標FARを固定したときの閾値とFRRを記録する。PoCの主条件は目標FAR 1%とし、0.1%も
補助値として記録する。両FAR目標の閾値はvalidation impostor scoreから固定し、その閾値を
testへそのまま適用する。

主閾値は母音別・seed別・登録数別に設定する。較正には主`verification`だけを用い、
cross-text・短区間・品質層・感度診断にも同じ閾値を使う。全母音共通のpooled閾値も同じ手順で
補助報告するが、主評価方式の選び直しには使わない。
FAR目標用の候補閾値は`-inf`と各unique impostor scoreの`nextafter(score, +inf)`とし、
経験FARが目標以下となる最小閾値を選ぶ。tieをランダムに受理しない。
EER用の運用閾値はgenuineとimpostorを合わせたunique scoreから同じ方法で候補を作り、
`abs(FAR-FRR)`最小、tieならFAR最小、さらにtieなら数値の小さい閾値を選ぶ。
無限値は成果物JSONでは文字列`"-inf"` / `"+inf"`で保存し、JSONの非標準NaN等を使わない。
各閾値の実測FARと誤受理数・impostor数を報告し、目標1% / 0.1%の達成がtestや実運用で
保証されるとは書かない。低FARの精度はtrialの依存と15話者という規模に制約される。

## 6. testによる最終評価

入力方式、前処理、encoder、checkpoint選択規則、登録区間数、スコア、閾値をvalidationで
確定した後、testの15話者で同じ登録・照合手順を一度実行する。
選択設定の70話者3 seedのcheckpointと、各checkpoint自身から較正した閾値を全て凍結bundleに
含める。testの3 seed個別値・平均・母標準偏差を報告し、最良seedだけを報告しない。
ensembleは行わず、単一配布モデルは事前指定seed `20260926`とする。学習曲線用の小cohortや
非採用設定をtestへ持ち込まない。testのenrollmentは同じ固定抽出規則で新たに作成する。
testでは主・cross-text・登録数・長さ・品質層の事前定義済み集計を一括実行する。
境界・gain感度はvalidation診断に限定する。

testでは次の二種類を区別して報告する。

- test内で曲線から算出するEER、ROC、DET: 閾値に依存しない最終的な分離性能
- validationで固定した閾値をそのまま適用したFAR、FRR: 閾値の一般化性能

testの結果を理由に入力方式や閾値を変更した場合、その結果を最終評価として扱わない。
変更後の方式はvalidationで再選択し、別の未使用評価データを用意する。

## 7. 保存する成果物

各実験について、最低限次を保存する。

- 実験ID、実行日時、Git commit
- train cohort、区間IDまたはmanifestのchecksum
- 乱数seed
- 入力特徴と正規化の設定
- モデル構造、embedding次元、損失関数
- sampler、optimizer、学習率、更新回数
- checkpointと選択理由
- 登録区間ID、照合trial一覧
- 話者・母音・役割・区間長を含む照合スコア
- 集計指標と曲線データ
- 実行時間、パラメータ数、メモリ使用量

設定、checkpoint、スコア、集計結果を同じ実験IDで追跡できる構成にする。

成果物はGit管理外の`artifacts/phoneme-speaker-encoder/<experiment_id>/`へ次の構成で保存する。
文書や集計値をリポジトリへ追加する場合も、元のexperiment IDとchecksumを記載する。

```text
<experiment_id>/
├── config.json
├── environment.json
├── data.json
├── feature-statistics.json
├── execution-budget.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── selections/
│   ├── enrollment-segments.jsonl
│   ├── trials.jsonl
│   ├── thresholds.json
│   └── frozen-evaluation-bundle.json
├── scores/
│   ├── validation.jsonl
│   └── test.jsonl
└── metrics/
    ├── validation.json
    └── test.json
```

上図は最終評価runを含む配置例であり、testファイルと凍結bundleは該当段階に達するまで作らない。
感度診断の派生trial・score・metricsは`diagnostics/`、選択順位表とbootstrapは比較全体の
`artifacts/phoneme-speaker-encoder/comparisons/<comparison_id>/`へ保存する。

experiment IDは`<UTC YYYYMMDDTHHMMSSZ>-<git short sha>-s<seed>-<config sha先頭8桁>`とする。
`config.json`はすべての確定パラメータを展開したJSON、`environment.json`はOS、Python、
framework、CUDA、device情報、依存ライブラリversion、`data.json`はmanifest、cohort ID、
前処理とtrain特徴統計のpath・SHA-256を持つ。特徴統計の実体も`feature-statistics.json`として
runへ保存し、runが異なる外部pathの上書きで壊れないようにする。生波形では統計をnullとする。

checkpointにはmodel、optimizer、scheduler、update、run seed、config SHA-256、manifest
SHA-256、特徴統計SHA-256を含める。加えてPython / NumPy / framework CPU・全deviceの乱数状態、
samplerの話者選出回数と次logical update、early stopping基準値・patience、best指標・update、
使用時のAMP scaler状態を保存する。保存はoptimizer stepとzero_gradの後だけとし、途中の
microbatchからは再開しない。prefetch済みbatchは破棄し、次logical updateのhashから再生成する。
入力cropをworkerの乱数消費に依存させないため、worker数・prefetch順に左右されない。
比較runでは不一致を停止させる。overrideする場合は別run IDを作り、継続再現とは扱わない。
score JSONLはtrial ID、split、role、speaker、claimed speaker、母音、区間ID、区間長、登録区間数、
cosine scoreを必須とする。test成果物は最終設定を固定するまで作成しない。
品質層の分析用に元segmentの属性とsource IDをjoin可能にする。変換trialには元trial ID、
変換量、除外理由を追加する。閾値一覧は`selections/thresholds.json`、採用候補・checkpoint hash・
3 seed・全評価条件は`selections/frozen-evaluation-bundle.json`、資源上限は
`execution-budget.json`へ保存する。採用順位表、bootstrapの抽出IDと区間、診断結果も保存する。

## 8. 小規模な学習成立性確認

本比較へ進む前に次を順に満たす。

1. 専用の固定32区間batchを過学習させ、500 update以内にtrain話者分類accuracy 95%以上となる。
2. 10話者cohortを2,000 update学習し、損失、gradient、embeddingが有限値を保つ。
3. enrollment profile、genuine/impostor trial、macro EERまでを一つのrun IDで生成できる。
4. 同じcheckpointとtrialで評価を2回実行し、scoreと指標が一致する。
5. 異なる長さを混ぜたbatchでpadding不変性テストが通る。
6. dropout無効で母音別microbatchと一括batchのloss・gradientが許容誤差内で一致する。
7. 同一環境で連続10 updateと5 update保存・再起動・残り5 updateの入力ID、crop、損失、
   model / optimizer / scheduler状態が一致する。worker数変更でも入力IDとcropが一致する。

過学習用は10話者cohortのID辞書順先頭4話者×a/i/u/e×2区間=32件とし、各組でsegment ID
辞書順先頭2件を使う。通常のP=10・5母音・100区間samplerは使わない。毎update全32件を使用し、
分類headは4 class、RMSあり、center crop、dropout=0、AAM+SupCon、AdamW lr=3e-4・weight decay=0、
constant schedule、early stoppingなしとする。4母音の8区間ずつを処理し、各損失を1/4で重み付けして
1 updateにまとめる。他の損失定数・gradient clip・float32は共通とする。
train accuracyはmarginを適用する前のcosine最大classで測る。特徴統計は10話者cohortの対応する
train統計を使う。6 encoderで1回ずつ行い、成立性確認以降は通常設定へ戻す。
単体テストでは5母音すべてと両lossのforward / backward、閾値のtie、全score同値、
品質層の欠損を網羅する。過学習用の4母音だけで実装網羅とみなさない。

成立性確認のEER自体は方式選択に使わない。失敗した場合はtestを実行せず、原因と変更した
設定を新しいrunとして記録する。

## 9. 後続Issueへの分割

実装は依存順に次のIssueへ分けられる。

1. **Dataset / DataLoaderと特徴統計**: source slice読出し、入力変換、mask、balanced sampler、
   train統計、enrollment/trial固定、単体テスト。
2. **encoderと学習**: MLP、TDNN、生波形CNN、pooling、AAM-Softmax、within-vowel SupCon、
   checkpoint、10話者成立性確認。
3. **登録・照合評価**: profile、cosine score、EER/FAR/FRR、区間長別集計、曲線、固定閾値。
4. **70話者比較実験**: 本比較16 + 限定比較2、3 seed、checkpoint選択、validationによる方式確定。
5. **学習曲線と診断**: 採用設定の10 / 25 / 50話者追加、70話者再利用、品質・境界・gain診断。
6. **test最終評価**: 選択済み設定の3 checkpointと閾値を凍結し、testを一度だけ評価して文書化。

各Issueは前段の成果物schemaを変更しない。変更が必要な場合はschema versionを上げ、既存runと
混在させない。

## 10. 設計上の保留事項

- 実運用で許容するFARは製品要件が定まった時点で決める。PoCでは1%を主条件とする。
- 資源上限とbenchmark APIは実行環境確定時に性能を見る前に登録する。母音別microbatch20でも
  実行できない場合は、候補だけを不公平に変更せず、次の事前登録版で共通条件を見直す。
- 固定したoptimizer、SupCon重み・temperature、128次元、母音条件付けなしの一般的最適性は未検証。
  全発話方式との比較はPhase 7、別日・別端末は外部評価段階に残す。
