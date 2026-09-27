# 生波形encoder設計

## 1. 目的

24 kHzの母音波形を直接入力し、log-Melで失われる位相や微細な周期構造を利用できるかを
検証する。生波形方式の中でも、短い初段kernelと長い初段kernelのどちらが有利かは事前に
一意に決めにくいため、その他の構造を揃えた2候補をvalidationで比較する。
短受容野だけで方式全体を判断しないため、長文脈を扱う第3候補も限定設定で検証する。

## 2. 入出力

- 入力: `float32[B, 1, N]`、24 kHz、`720 <= N <= 6000`
- waveform mask: `bool[B, N]`
- 前処理: 区間内DC除去。探索空間に従い上限付きRMS正規化あり / なしを比較する
- padding: batch内最大長まで末尾を0埋め
- 出力: L2正規化した`float32[B, 128]` speaker embedding
- 母音ラベル: encoderへ入力しない

RMS正規化後にpeak clippingは行わない。浮動小数点波形が`[-1, 1]`を超えても保持し、clippingに
よる非線形歪みを追加しない。データセット全体の波形平均・標準偏差による正規化やpre-emphasisは
使用しない。

## 3. 共通構造

本比較の2候補は第1層のkernel sizeだけを変え、残りを共通化する。

| 段 | channel | kernel | stride | padding |
| --- | --- | ---: | ---: | --- |
| Conv 1 | 1→64 | 候補ごとに80または240 | 4 | 0 |
| Conv 2 | 64→128 | 8 | 4 | 0 |
| Conv 3 | 128→192 | 5 | 2 | 0 |
| Conv 4 | 192→256 | 3 | 2 | 0 |

各blockは`Conv1d(bias=false) -> channel方向LayerNorm(eps=1e-5) -> GELU -> dropout 0.1`
とする。畳み込みはすべてvalidとし、波形区間外の値を受容野へ加えない。block間のresidual接続は
channel数と時間長が変わるため使用しない。

最終時系列をmask付き平均と標準偏差でpoolingし、TDNN候補と同じprojectionを使う。

```text
masked mean + masked std（512次元）
  -> Linear 512→256
  -> GELU
  -> dropout 0.1
  -> Linear 256→128
  -> L2 normalization
```

標準偏差は`sqrt(max(population_variance, 1e-5))`、L2正規化は`x / max(norm(x), 1e-12)`
とする。projectionのLinearはbiasありとする。AAM-Softmaxのclassification headはencoderに
含めず、学習時だけ外付けする。

## 4. 比較候補

| ID | Conv 1 kernel | Conv 1時間幅 | 最終受容野 | 30 ms出力長 | 250 ms出力長 | encoder概算parameter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `waveform_cnn_k80` | 80 sample | 3.33 ms | 236 sample / 9.83 ms | 8 | 91 | 506,496 |
| `waveform_cnn_k240` | 240 sample | 10.00 ms | 396 sample / 16.50 ms | 6 | 88 | 516,736 |
| `waveform_cnn_k240_context27`（限定） | 240 sample | 10.00 ms | 2,060 sample / 85.83 ms | 6 | 88 | 524,160 |

`waveform_cnn_k80`は周期内の局所構造を段階的に構成する候補、`waveform_cnn_k240`は低い
基本周波数の約1周期を初段から観測できる候補である。どちらが短母音の話者差に有利かを
事前判断せず、同じtrialのvalidation macro EERで選択する。

参考としてlog-Mel TDNN候補は約411,136 parameterである。生波形候補は約23〜26%大きいが、
front-endの入力channel差に伴う範囲として許容し、parameter数、推論時間、最大メモリを結果に
併記する。分類headはparameter比較に含めない。

### 4.1 長文脈の限定候補

`waveform_cnn_k240_context27`はk240のConv 4とpoolingの間へ、depthwise Conv1d
256→256、groups=256、kernel=27、stride=1、dilation=1、padding=13、bias=falseを1層追加する。
channel LayerNorm、GELU、dropout 0.1、block後のmaskは他blockと共通とする。
追加parameterは`256 × 27 + 2 × 256 = 7,424`。時系列長とprojectionは変わらない。
Conv 4のframe間隔は64 sampleなので、受容野は`396 + 26 × 64 = 2,060 sample`となる。

このcontext層だけはsame zero paddingを使う。層の前後で無効位置を0にし、元区間外の
実音声を読まない。最短区間でも6 frameを維持するが、実際に観測できる音声は区間長が上限であり、
85.83 msの実音声を30 ms区間から作るわけではない。

log-Mel TDNNの公称受容野は`25 + (2×1 + 2×2 + 2×3) × 5 = 85 ms`である。
長文脈候補はこの時間範囲を探索に含めるためのもので、同一情報量・同一構造の保証ではない。
k240との差には追加非線形層とpaddingも含まれるため、受容野だけの因果効果とは解釈しない。

## 5. maskと出力長

各sampleの有効長を`L_0`とし、各valid convolution後の長さを次で更新する。

```text
L_i = floor((L_(i-1) - kernel_i) / stride_i) + 1
```

各blockの直後に`position < L_i`をmaskとし、無効位置を0へ戻す。validな出力位置の受容野は
すべて元の有効波形内に収まるため、同じ区間を単独で処理した場合と長い区間を含むbatchで
処理した場合の有効出力が一致する。

最短720 sampleに対する層ごとの長さは次のとおりである。

- `waveform_cnn_k80`: `720 -> 161 -> 39 -> 18 -> 8`
- `waveform_cnn_k240`: `720 -> 121 -> 29 -> 13 -> 6`

k240_context27はk240と同じ長さの後に、6→6（最長88→88）のcontext層を持つ。
一般の出力長式は`floor((L + 2p - d(k-1) - 1) / s) + 1`とする。

全候補で最終poolingに複数時刻を残す。入力長が720 sample未満の場合や、いずれかの層で
有効長が1未満となる場合は実験を停止する。

## 6. 比較条件

次は3候補とlog-Mel候補で共通化する。

- 128次元embeddingとmask付きmean / std pooling
- batch sampling、optimizer、最大更新数と停止規則、seed（限定候補はAAMのみ）
- enrollment区間、verification trial、checkpoint選択規則
- 250 ms cropと30 ms最小長

生波形候補では`RMS正規化あり / なし`、2種類のkernel、2種類のlossを直積し、8通りを
評価する。kernel以外のchannel、stride、activation、normalizationを同時に変更しない。
これとは別にk240_context27をRMSあり・AAMのみの1設定、3 seedで学習し、同条件のk240を
対照として再利用する。限定候補も採用対象に含めるが、他のRMS・loss条件へ結果を外挿しない。

## 7. 実装テスト

- 720 sampleと6,000 sampleで、3候補の各層出力長が設計値と一致する。
- batch末尾のpadding量やpadding値を変えても、有効位置とembeddingが許容誤差内で一致する。
- mask付きpoolingが無効位置を平均と標準偏差へ含めない。
- 720 sample入力のforward / backwardでNaN、Inf、空maskが発生しない。
- 同じseed、入力、checkpointから同じembeddingを再生成できる。
- parameter数を設定値と照合し、差がある場合は学習前に設定と文書を更新する。
- context層のmaskにより、短区間と長区間の混合batchでもpadding不変性が成立する。
