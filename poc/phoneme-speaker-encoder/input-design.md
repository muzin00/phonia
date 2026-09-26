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

## 3. 可変長入力の基本方針

母音区間を、単純な波形の伸縮によって一律のサンプル数へ変換しない。伸縮すると時間尺度と
周波数が変わり、話者特徴とは無関係な変形を入力へ加えるためである。

第一候補では元の区間長を維持して音響特徴を生成し、batch内だけpaddingする。各入力の
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

最小入力長はPhase 2で確定した30 msを当面維持する。短区間を追加で除外するかどうかは、
母音ごとのデータ量を変えるため、学習後の区間長別評価なしに決めない。

長区間の切り詰め上限も現時点では固定しない。全長入力と上限付き入力を同じ条件で学習・
評価し、性能と計算量から決定する。

## 4. 比較する入力方式

### 4.1 最小ベースライン

波形からlog-Melを生成し、時間方向の平均と、必要に応じて標準偏差を求め、MLPで固定長
embeddingへ変換する。

時間方向の変化をほとんど利用しない構成であり、学習パイプラインの成立性と、時間方向の
encoderを追加する価値を確認する基準とする。

### 4.2 第一候補: 可変長log-Mel

可変長log-Melを小型のCNNまたはTDNNへ入力し、mask対応poolingで固定長embeddingへ
変換する。

窓長、hop長、Mel bin数、端点処理は設定として保存する。初期比較では少なくとも25 ms窓の
5 ms hopと10 ms hopを候補とする。これらは学習結果によって選択し、現時点の最終仕様とは
しない。

### 4.3 比較候補: 可変長生波形

元波形を1D CNNなどへ入力し、時間方向のpoolingで固定長embeddingへ変換する。
log-Melで失われる情報を利用できる可能性を検証する一方、最短720サンプルの入力でも
成立する受容野とstrideを設計する。

log-Mel方式と完全に同じfront-endにはできないため、backend、embedding次元、更新回数、
おおよそのパラメータ数を可能な範囲で揃え、学習時間、推論時間、メモリも併記する。

### 4.4 補助比較: 固定長化

必要に応じて、短区間の反復と長区間のcropによる固定長入力を比較対象へ加える。反復は
新しい音声情報を増やさず、人工的な継ぎ目を作るため、第一候補にはしない。

ゼロpaddingを固定長入力として使う場合も、有効長またはmaskを必須とする。paddingを
含めてpoolingや正規化を行う方式は比較対象にしない。

## 5. 入力の正規化

入力の正規化は次の原則に従う。

- 波形を浮動小数点へ変換し、DC成分を除去する。
- 音量差を話者識別の手掛かりにしないため、上限付きRMS正規化を初期候補とする。
- RMSの下限を設け、低レベル信号を過度に増幅しない。
- log-Melの帯域別平均・標準偏差はtrainだけから算出し、validationとtestへ同じ値を使う。
- 母音区間ごとに帯域別時間平均を引く正規化は、平均スペクトルに含まれる話者差も消す
  可能性があるため初期方式に入れない。
- 正規化方法も入力方式の一部として設定と成果物へ保存する。

RMS正規化の有無と値は学習後に比較して確定する。validationやtestから正規化統計を
算出しない。

## 6. 未決事項

- log-Melの確定パラメータと端点処理
- 生波形方式を比較に含める段階とモデル構造
- 入力長の上限と長区間のcrop方法
- RMS正規化の確定方式
- paddingとmaskの具体的なtensor表現
