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

`phase3-vowel-segments.jsonl`の採用区間を集計した結果を、入力上限と正規化の根拠にする。

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
| waveform | mono、`float32`、PCMを`[-1, 1]`へ変換 |
| DC除去 | 区間内の波形平均を減算 |
| RMS正規化 | 目標`-26 dBFS`、gainを`[-12, +12] dB`へclamp |
| STFT窓 | 600 sample（25 ms）のperiodic Hann窓 |
| hop | 120 sample（5 ms） |
| FFT | 600点、power spectrum、`center=false` |
| Mel filterbank | 64 bin、50–12,000 Hz、Slaney scale / normalization |
| log | `ln(max(mel_power, 1e-10))` |
| 特徴正規化 | train全frameから求めたMel bin別平均・標準偏差 |
| 最小長 | 720 sample（30 ms）。未満はエラーにする |
| 最大長 | 6,000 sample（250 ms） |
| 長区間 | trainはrandom crop、validation/testはcenter crop |
| tensor | 特徴`float32[B, 64, T]`、mask`bool[B, T]` |
| padding | batch内の最大`T`まで末尾を0埋め |

RMSはDC除去後の波形から求める。無音に近い値でも除算が不安定にならないよう、RMS計算時の
振幅下限を`1e-5`とする。gainのclampにより、小音量区間を無制限に増幅しない。

250 msは全採用区間の99.9%を切り詰めずに扱える上限である。crop開始位置に使う乱数は
DataLoader workerのseedから導出し、同一runを再現できるようにする。validation/testでは
常に同じcenter cropを使う。

25 ms窓と5 ms hopでは、最短30 ms区間から2 frameを得る。`center=false`とし、区間外の
反射paddingを音声として追加しない。特徴を区間ごとに生成してからbatch化するため、同じ
区間の特徴はbatch内の他区間の長さに依存しない。

## 5. 可変長入力の処理

母音区間を、単純な波形の伸縮によって一律のサンプル数へ変換しない。伸縮すると時間尺度と
周波数が変わり、話者特徴とは無関係な変形を入力へ加えるためである。

参照方式では元の区間長を維持して音響特徴を生成し、batch内だけpaddingする。各入力の
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

長さの近い区間を同じbatchへ配置するbucketingは、padding量を減らす候補とする。ただし、
話者と母音を均衡させるsamplingを優先し、その制約を壊さない範囲で使用する。

最小入力長はPhase 2で確定した30 msを維持する。短区間を追加で除外するかどうかは、
母音ごとのデータ量を変えるため、学習後の区間長別評価なしに決めない。

最大長は250 msに固定する。全採用区間の99.9%を切り詰めずに扱えるため、切り詰めなしの
候補を追加しても得られる情報に対して組み合わせ数と計算量の増加が大きい。

## 6. 事前登録する比較

### 6.1 最小ベースライン

波形からlog-Melを生成し、時間方向の平均と、必要に応じて標準偏差を求め、MLPで固定長
embeddingへ変換する。

時間方向の変化をほとんど利用しない構成であり、学習パイプラインの成立性と、時間方向の
encoderを追加する価値を確認する基準とする。

### 6.2 可変長log-Mel候補

可変長log-Melを小型のCNNまたはTDNNへ入力し、mask対応poolingで固定長embeddingへ
変換する。

25 ms窓、5 ms hop、64 Mel binに固定する。10 ms hopでは最短30 ms区間が1 frameとなり、
短い区間が多い今回のデータで時間方向の情報をさらに減らすため、比較候補へ含めない。

### 6.3 可変長生波形候補

元波形を1D CNNなどへ入力し、時間方向のpoolingで固定長embeddingへ変換する。
log-Melで失われる情報を利用できる可能性を検証する一方、最短720サンプルの入力でも
成立する受容野とstrideを設計する。

log-Mel方式と完全に同じfront-endにはできないため、backend、embedding次元、更新回数、
おおよそのパラメータ数を可能な範囲で揃え、学習時間、推論時間、メモリも併記する。

### 6.4 補助比較

必要に応じて、短区間の反復と長区間のcropによる固定長入力を比較対象へ加える。反復は
新しい音声情報を増やさず、人工的な継ぎ目を作るため、第一候補にはしない。

ゼロpaddingを固定長入力として使う場合も、有効長またはmaskを必須とする。paddingを
含めてpoolingや正規化を行う方式は比較対象にしない。

比較対象は次のとおりとする。

1. log-Mel最小ベースラインでデータ読み出しから照合評価までを成立させる。
2. log-Melについて、`RMS正規化あり / なし`、`統計pooling + MLP / TDNN`、
   `AAM-Softmax / AAM-Softmax + 母音内SupCon`の8通りを評価する。
3. 可変長生波形について、`RMS正規化あり / なし`と2種類のlossを組み合わせた4通りを
   評価する。

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

## 8. 実装時の検証項目

- 30 ms、250 ms、250 ms超の区間で期待したframe数とcrop位置になる。
- 同じ区間の特徴が、単独入力と異なる長さを含むbatch入力で一致する。
- padding値を変えても有効frameのencoder出力と最終embeddingが変化しない。
- train統計のchecksumが一致しない場合はcheckpointを読み込まない。
- NaN、Inf、空のmask、0除算を検出して実験を停止する。
