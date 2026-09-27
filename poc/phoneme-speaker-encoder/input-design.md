# 入力設計

## 1. 目的

母音1区間を共通encoderへ入力する方法と、長さの異なる区間をbatchで扱う方法を定義する。
入力方式は学習前の特徴統計だけでは確定せず、同じ条件でencoderを学習した結果から選択する。

## 2. 入力データ

入力の正本は次のmanifestとJVS元音声である。

- manifest: `poc/phoneme-speaker-dataset/data/generated/phase3-vowel-segments.jsonl`
- 音声: 各レコードの`source_file`
- 区間: `start_frame:end_frame`
- 形式: 24 kHz、モノラル、16-bit PCM
- 採用区間: `near_silent`を固定規則で除外した462,242区間

以下の手順やrun成果物で使う`segment_id`は元manifestの`vowel_interval_id`、`vowel`は
`normalized_phoneme`の別名とする。元レコードのIDを再採番せず、mappingをrun設定に保存する。

分割と用途は次のとおりである。

| 分割 | 話者数 | 区間数 | 用途 |
| --- | ---: | ---: | --- |
| train | 70 | 323,913 | encoderのパラメータ更新 |
| validation | 15 | 69,220 | 入力方式、モデル、checkpoint、学習設定、閾値の選択 |
| test | 15 | 69,109 | 全設定を固定した後の最終評価 |

train、validation、testの間で話者を共有しない。validationとtestの音声ではencoderを
更新しない。testの結果を見て入力方式や設定を選択しない。

データの生成規則と役割分割は、次を正本とする。

- [JVSデータセット設計](../phoneme-speaker-dataset/dataset-design.md)
- [生成・品質確認結果](../phoneme-speaker-dataset/dataset-results.md)
- [入力設定](../phoneme-speaker-dataset/config/phase3-input.json)
- [入力検証結果](../phoneme-speaker-dataset/data/phase3-vowel-dataset-validation.json)

## 3. 区間長と音量の分布

次は以前に`phase3-vowel-segments.jsonl`の全splitを集計した記述統計である。
testの長さメタデータも含む既知の集計であり、完全に未閲覧のデータとは表現しない。
モデルscore・認証性能は参照していない。以後の統計再生成と前処理判断はtrainだけで行い、
この表を再集計してtestに合わせて入力条件を調整しない。

| 対象 | 最小 | 中央値 | p95 | p99 | p99.9 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全区間 | 30 ms | 70 ms | 150 ms | 190 ms | 250 ms | 1,510 ms |
| `/a/` | 30 ms | 80 ms | 150 ms | 190 ms | - | 560 ms |
| `/i/` | 30 ms | 60 ms | 140 ms | 180 ms | - | 660 ms |
| `/u/` | 30 ms | 50 ms | 120 ms | 170 ms | - | 1,510 ms |
| `/e/` | 30 ms | 80 ms | 160 ms | 200 ms | - | 310 ms |
| `/o/` | 30 ms | 70 ms | 160 ms | 210 ms | - | 1,040 ms |

train区間のRMSは中央値`-26.087 dBFS`、p01が`-45.647 dBFS`、p99が
`-17.228 dBFS`である。統計値を再生成する処理はtrainだけを読み、集計対象manifestの
SHA-256と集計コードのGit commitを成果物へ保存する。

## 4. 参照入力候補

次の値を最初に実装する参照候補とする。これは最終仕様ではない。判断が難しい項目は後述の
候補と組み合わせて学習し、validation性能から採用値を決める。各組み合わせは独立した設定と
実験IDを持ち、実験開始後に値を上書きしない。

| 項目 | 値 |
| --- | --- |
| sample rate | 24,000 Hz。異なる入力はエラーにする |
| waveform | mono、`float32`、signed int16 PCMを32768で除算 |
| DC除去 | 区間内の波形平均を減算 |
| RMS正規化 | 目標`-26 dBFS`、gainを`[-12, +12] dB`へclamp |
| STFT窓 | 600 sample（25 ms）のperiodic Hann窓 |
| hop | 120 sample（5 ms） |
| FFT | 600点、power spectrum、`center=false` |
| Mel filterbank | 64 bin、50–12,000 Hz、Slaney scale / normalization |
| log | `ln(max(mel_power, 1e-10))` |
| 特徴正規化 | 使用cohort・RMS条件ごとにtrainの固定center cropから求めたMel bin別統計 |
| 最小長 | 720 sample（30 ms）。未満はエラーにする |
| 最大長 | 6,000 sample（250 ms） |
| 長区間 | trainはrandom crop、validation/testはcenter crop |
| tensor | 特徴`float32[B, 64, T]`、mask`bool[B, T]` |
| padding | batch内の最大`T`まで末尾を0埋め |

処理順はsource slice読出し→PCM変換→crop→DC除去→任意のRMS正規化→特徴生成→
train統計で特徴正規化→batch paddingとする。DCとRMSはcrop後の有効波形だけから求める。
RMSはDC除去後の波形から求める。無音に近い値でも除算が不安定にならないよう、RMS計算時の
振幅下限を`1e-5`とする。gainのclampにより、小音量区間を無制限に増幅しない。正規化後の
peak clippingは行わず、log-Melと生波形の両方式へ同じ浮動小数点波形を渡す。

250 msは既知の全採用区間の99.9%を切り詰めずに扱える予算上限であり、長区間の影響が
認証性能上無視できると実証したわけではない。crop対象の件数と条件別性能を報告する。
train crop開始位置は学習・評価設計のhash規則で
`["crop", run_seed, logical_update, segment_id]`をhashし、その256-bit非負整数を
`(original_length - 6000 + 1)`で割った余りとする。6,000以下ならcropしない。
worker seedやprefetch順に依存させない。評価と特徴統計計算は開始位置
`floor((original_length - 6000) / 2)`のcenter cropを使う。

25 ms窓と5 ms hopでは、最短30 ms区間から2 frameを得る。`center=false`とし、区間外の
反射paddingを音声として追加しない。特徴を区間ごとに生成してからbatch化するため、同じ
区間の特徴はbatch内の他区間の長さに依存しない。

## 5. 可変長入力の処理

母音区間を、単純な波形の伸縮によって一律のサンプル数へ変換しない。伸縮すると時間尺度と
周波数が変わり、話者特徴とは無関係な変形を入力へ加えるためである。

参照方式では250 ms以下の元の区間長を維持して音響特徴を生成し、batch内だけpaddingする。各入力の
有効長またはmaskを保持し、encoderと時間方向のpoolingがpaddingを話者特徴に使用しない
構成にする。

```text
可変長の母音波形
        ↓
可変長の音響特徴列
        ↓ batch内padding + 有効長/mask
共通encoder
        ↓ mask対応の時間方向pooling
固定長speaker embedding
```

paddingの影響は最終poolingだけで除けばよいとは限らない。時間方向を混合する畳み込み、
attention、正規化を使用する場合は、各層で無効位置が有効位置へ影響しないことを実装テスト
で確認する。音響特徴は区間ごとに生成してからbatch化し、STFTの端点処理がbatch内の
最大長によって変わらないようにする。

このversionではbucketingを使用しない。母音単位のmicrobatchとmaskで扱い、話者と母音を
均衡させるsamplingを優先する。将来導入する場合も抽出区間集合と論理batch構成を変えない。

最小入力長はPhase 2で確定した30 msを維持する。短区間を追加で除外するかどうかは、
母音ごとのデータ量を変えるため、学習後の区間長別評価なしに決めない。

最大長は今回の予算上250 msに固定する。切り詰めなしの改善余地は未検証として残し、
250 ms超のqueryを別集計する。診断結果を見て同じ探索のcrop上限を変更しない。

## 6. 事前登録する比較

### 6.1 最小ベースライン

波形からlog-Melを生成し、時間方向の平均と標準偏差を求め、MLPで固定長
embeddingへ変換する。

時間順序を利用しない構成であり、学習パイプラインの成立性と性能・コストの基準とする。
TDNNとはparameter数が約6.24倍違うため、時間混合の限定比較には別途、規模を約0.19%差へ
揃えた`framewise_cnn`を使う。kernel=1のためpooling前に時刻間の混合を行わない。

### 6.2 可変長log-Mel候補

可変長log-Melを小型のCNNまたはTDNNへ入力し、mask対応poolingで固定長embeddingへ
変換する。

25 ms窓、5 ms hop、64 Mel binに固定する。10 ms hopでは最短30 ms区間が1 frameとなり、
短い区間が多い今回のデータで時間方向の情報をさらに減らすため、比較候補へ含めない。

### 6.3 可変長生波形候補

元波形を1D CNNへ入力し、時間方向のpoolingで固定長embeddingへ変換する。
log-Melで失われる情報を利用できる可能性を検証する一方、最短720サンプルの入力でも
成立する受容野とstrideを設計する。

初段kernelが80 sampleと240 sampleの2候補を比較し、それ以外のchannel、stride、pooling、
projectionを共通化する。詳細は[生波形encoder設計](waveform-encoder-design.md)を正本とする。
追加の長文脈候補`waveform_cnn_k240_context27`はRMSあり・AAMのみで限定比較する。
log-Mel方式と完全に同じfront-endにはできないため、embedding次元、最大更新数、停止規則を
揃え、parameter数、受容野、学習時間、推論時間、メモリの差を併記する。
結果は構成同士の比較であり、入力方式一般の優劣と解釈しない。

### 6.4 現在の探索から除外する方式

短区間の反復と長区間のcropによる固定長入力は現在の探索へ含めない。反復は新しい音声情報を
増やさず、人工的な継ぎ目を作るためである。将来追加する場合は、現在の結果を見て同じ探索へ
後付けせず、別versionの探索空間として事前登録する。

ゼロpaddingを固定長入力として使う場合も、有効長またはmaskを必須とする。paddingを
含めてpoolingや正規化を行う方式は比較対象にしない。

比較対象は次のとおりとする。

1. log-Mel最小ベースラインでデータ読み出しから照合評価までを成立させる。
2. log-Melについて、`RMS正規化あり / なし`、`統計pooling + MLP / TDNN`、
   `AAM-Softmax / AAM-Softmax + 母音内SupCon`の8通りを評価する。
3. 可変長生波形について、`RMS正規化あり / なし`、`80 / 240 sample初段kernel`、
   2種類のlossを組み合わせた8通りを評価する。
4. 規模を揃えた時間混合なしモデルと長文脈生波形モデルを各1設定で追加比較する。
   全18設定で採用候補を選び、採用構成だけの学習曲線と再学習不要の診断を行う。

単一項目の効果だけでなく、項目間の相互作用も評価対象とする。比較対象の直積と除外する
非互換な組み合わせは、学習開始前に設定一覧として固定する。

## 7. 入力の正規化

入力の正規化は次の原則に従う。

- 波形を浮動小数点へ変換し、DC成分を除去する。
- 音量差を話者識別の手掛かりにしないため、上限付きRMS正規化を参照候補とする。
- RMSの下限とgainの上下限を設け、低レベル信号を過度に増幅しない。
- log-Melの帯域別平均・標準偏差はtrainだけから算出し、validationとtestへ同じ値を使う。
- 母音区間ごとに帯域別時間平均を引く正規化は、平均スペクトルに含まれる話者差も消す
  可能性があるため参照候補に入れない。
- 正規化方法も入力方式の一部として設定と成果物へ保存する。

RMS正規化ありとなしを候補として比較する。validationやtestから正規化統計を算出しない。

### 7.1 特徴統計の計算と適用範囲

cohort（10 / 25 / 50 / 70）とRMSあり / なしの組ごとに、該当するtrain・training区間を
segment ID辞書順に1回ずつ読み、固定center crop後の前処理で統計を計算する。
有効な全frameを同じ重みとし、Mel bin別にfloat64で母平均・母分散を集計する。
正規化式は`(feature - mean) / max(sqrt(population_variance), 1e-5)`とする。
本学習時のrandom cropは統計計算時には使わず、この参照分布との差も既知の設計条件とする。

同じcohort・入力前処理ならencoder、loss、seed間で同一統計を再利用する。
10 / 25 / 50話者へ70話者の統計を流用しない。validation / testには対応する学習runの統計を
そのまま適用する。生波形はdataset全体の特徴統計を使用しない。
統計にはcohortの話者・区間ID checksum、manifest SHA-256、前処理config SHA-256、件数、
frame数、平均・分散の実体、計算コードのcommitを保存する。

## 8. 実装時の検証項目

- 30 ms、250 ms、250 ms超の区間で期待したframe数とcrop位置になる。
- 同じ区間の特徴が、単独入力と異なる長さを含むbatch入力で一致する。
- padding値を変えても有効frameのencoder出力と最終embeddingが変化しない。
- train統計のchecksumが一致しない場合はcheckpointを読み込まない。
- NaN、Inf、空のmask、0除算を検出して実験を停止する。
- worker数・prefetch・再起動を変えても同じseed・logical updateの区間IDとcropが一致する。
- 各cohortの特徴統計に、cohort外のtrain話者やvalidation / testが混入しない。
