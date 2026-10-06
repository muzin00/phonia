# 全validationのscore生成・閾値校正結果

後続の実行凍結・固定閾値test・全条件CIも完了した。[最終比較結果](evaluation-results.md)を参照。

2026-10-07に比較設計2.0.0の全validationを実行した。
15話者・1,200query、135個の登録profile、810,000件の照合記録を生成した。
方式・登録数・共通波形の長さ・supportごとの90条件で、計270個の閾値を校正した。
境界内方式の全45profile・full score 54,000件・full/nativeの9動作点はPhase 6と一致した。
モデルと特徴統計を更新せず、test音声・scoreは読み込んでいない。

この結果はvalidationでの閾値校正と診断である。
主比較の登録10・full・testの全入力FRR、speaker bootstrapの信頼区間はまだ算出していない。

## 固定条件と実行検査

- [比較protocol](config/comparison-protocol.json): 2.0.0。
  SHA-256 `5169ea422bf53ca11d52b47591befabaed8def7ee2d238de3ef9a0b61167c4ba`。
- [validation設定](config/validation.json): 全15話者、通常750query・別テキスト450query。
  各queryを本人1・他人14と比較し、登録数1・5・10の54,000個のPhase 6 trial IDを保持した。
- 3方式×full・最大1・2・3・5秒で810,000件に展開した。
  数値scoreは587,430件、母音不足の`no_score`は222,570件。元WAVは1,697個。
- 登録profileを全長さ条件で固定し、母音方式は同じanchor集合を使用した。
  ECAPAの登録には選択anchorを含む497個の固有元WAVを使用し、WAV単位の等重み平均とした。
- 母音方式はPhase 6のtorch 2.14.1、ECAPAはtorch/torchaudio 2.11.0・SpeechBrain 1.1.1。
  両方Python 3.12.13・NumPy 2.5.3、CPU float32、batch 1、intra/inter-op各1thread。
  L2正規化・平均・cosine・指標はfloat64で計算した。
- 音声cacheは16元WAVに制限し、scoreを圧縮JSONLへ逐次保存した。
  実入力frameとsource・modelのchecksumによるembedding cacheはpilotと同じ規則を使用した。

| 検査 | 結果 |
|---|---|
| 全件用workerとpilotの回帰検査 | 6個のembedding/profileファイルと3,780件のscore・欠測状態が完全一致 |
| 境界内の登録profile | 全45個でPhase 6とchecksum一致 |
| 境界内のfull score | 全54,000件で状態・数値が一致、最大絶対差0 |
| 公開API照合 | 全query・長さ条件の登録10で本人1・他人1、計12,000件が完全一致 |
| 固有母音embedding | 82,777個、128次元、有限・単位vector |
| 固有ECAPA embedding | 6,154個、192次元、有限・単位vector |
| 3回反復 | 全固有embeddingでbitwise一致 |
| 元音声・モデル・特徴統計 | 前後で不変、母音の特徴正規化tensorも不変 |
| 全照合記録 | ID・ラベル・query境界・profile固定・coverageの検査を通過 |
| 境界内full/nativeの閾値 | 登録数×3動作点の9個でPhase 6と一致 |
| 実装テスト | 47テスト、Ruff lint・format検査を通過 |

## 閾値と入力集合

nativeと共通入力について、3方式×登録数1・5・10×5長さ条件の計90条件を校正した。
閾値はvalidation / verificationのscored trialだけで作り、FAR 1%・0.1%・EER動作点の3個を保存した。
Phase 3の同点処理とfloat64 `nextafter`規則を用い、`score >= threshold`で受け入れる。
各FAR較正値が目標以下であることを独立に検査した。
別テキスト発話へ同じ閾値を適用し、別テキストの結果から較正し直していない。

共通入力は同じ長さ条件で5母音が揃うqueryの集合から決め、score値で選ばない。
ECAPAも同じ集合に絞った閾値を別途作成した。
共通入力の件数と元query集合に対する選択率は記録し、nativeのcoverageと区別した。
元発話5秒以上の固定集合は通常712・別テキスト177queryで、全capで同じqueryを使用した。
この集合の診断には各capのnative閾値を固定適用し、集合内で再較正していない。

最大1秒の母音方式では通常発話のscored impostorは658件で、FAR分解能は約0.152%。
目標FAR 0.1%の較正は、この条件では誤受入れ0件を要求する。
目標値以下の較正結果から、未知データのFAR 0.1%が保証されるとは判断しない。

## full・登録10のvalidation診断

以下はnative、通常発話で較正した目標FAR 1%の閾値を使用する。
全入力の本人分母は通常750・別テキスト450、他人分母は通常10,500・別テキスト6,300。
母音不足の本人queryは全入力FRRで拒否へ数える。
EERはscored trialのみを使用する。

| 方式 | role | coverage | 条件付きFAR | 条件付きFRR | 全入力FAR | 全入力FRR | EER |
|---|---|---:|---:|---:|---:|---:|---:|
| 境界内母音 | 通常 | 98.00% | 0.991% | 0.272% | 0.971% | 2.267% | 0.476% |
| 境界内母音 | 別テキスト | 87.11% | 0.784% | 1.020% | 0.683% | 13.778% | 0.948% |
| 文脈20 ms | 通常 | 98.00% | 0.991% | 0.272% | 0.971% | 2.267% | 0.544% |
| 文脈20 ms | 別テキスト | 87.11% | 0.802% | 0.255% | 0.698% | 13.111% | 0.711% |
| ECAPA | 通常 | 100.00% | 1.000% | 0.000% | 1.000% | 0.000% | 0.000% |
| ECAPA | 別テキスト | 100.00% | 0.635% | 0.000% | 0.635% | 0.000% | 0.111% |

通常発話は閾値較正にも使用している。表の0%はこのvalidation集合での観測値である。
学習データ・モデル規模・内部利用音声量は方式間で異なり、同じ登録元WAV集合とquery波形を使う
固定システムの比較として扱う。音素区間の効果だけを因果的に切り分けた結果とはしない。

## 入力長の診断（登録10）

長さ条件ごとに通常発話で閾値を作り、その条件の別テキストへ固定適用した。
登録profileは固定し、queryだけを共通中心の波形へ制限した。
同じ元発話を使用し、元発話が指定上限より短い場合は実長のまま入力した。

| 最大長 | 母音coverage：通常 / 別テキスト | 境界内全入力FRR：通常 / 別テキスト | 文脈全入力FRR：通常 / 別テキスト | ECAPA全入力FRR：通常 / 別テキスト |
|---|---:|---:|---:|---:|
| 1秒 | 6.27% / 6.22% | 93.87% / 94.22% | 93.73% / 94.00% | 14.53% / 15.11% |
| 2秒 | 32.67% / 42.67% | 67.73% / 59.11% | 67.73% / 58.00% | 1.20% / 1.11% |
| 3秒 | 66.80% / 72.00% | 33.87% / 28.67% | 33.33% / 28.67% | 0.27% / 0.00% |
| 5秒 | 90.80% / 84.89% | 9.60% / 16.00% | 9.60% / 15.56% | 0.00% / 0.00% |
| full | 98.00% / 87.11% | 2.27% / 13.78% | 2.27% / 13.11% | 0.00% / 0.00% |

ECAPAのcoverageはすべて100%。1秒条件は推論に成功した一方で、nativeのFRRが高かった。
1〜3秒の別テキストではECAPAの条件付きFARが1.25〜1.30%となり、較正目標1%を上回った。
これらも再較正せず保存した。

共通入力だけの通常発話では、ECAPAの登録10・目標FAR 1%のFRRは全長さ条件で0%だった。
ただし1秒の共通集合は47queryで、nativeの750queryと入力集合・較正閾値が異なる。
nativeと共通入力の結果を両方保持し、少数の入力充足queryだけの結果を全入力の性能へ一般化しない。

元alignmentは全長発話からの固定metadataを使っている。
短い録音からの再alignment・母音抽出精度、録音開始からの待ち時間は評価していない。

## 評価不能条件の保持

最大1秒の別テキストでは、母音方式のjvs016・jvs075・jvs079にscored queryがない。
nativeの母音2方式×3登録数の6条件と、共通入力の全3方式×3登録数の9条件について、
全15話者の条件付きFAR・FRR・EER・ROC/DETを評価不能として保存した。
母音不足による全拒否と全入力の分母は保持し、上表の全入力FRRへ反映した。
共通集合の条件にも、元query集合に対する選択率と各話者の件数を併記した。
全180評価条件のうち15条件でこの扱いとなった。

## 成果物と次の段階

runは`artifacts/phoneme-baseline-comparison/validation-20261007-v2`。
上書きせず、方式別のembedding・入力区間・profile・圧縮score、90条件の閾値、
180条件のcoverage・全動作点・ROC/DETデータ・5秒以上固定集合の診断を保存した。
各query・Phase 6 trial・model・profile・実入力frameをchecksum付きの記録から追跡できる。
独立検査は`validation-checks.json`へ保存した。

| 成果物 | SHA-256 |
|---|---|
| `validation-report.json` | `a1434a33b43e282f662aa309821c83d28c0ce250eb54f6974a3f6d9b82d77426` |
| `validation-thresholds.json` | `03999b8b744e5fc2d05778b04f0e6ea3dd55971ab4a9e1acbd0cd6b055f75b89` |
| `validation-metrics.json` | `f81cbd7176e5acc58f60c0b8ce28d16af9ec47fca592546447316895386828f1` |
| `validation-checks.json` | `66312ff2d6e7f366ca2f3e76a4726bf180844ea003ee14003c80a3c9718bfdd8` |

母音workerは約149秒、ECAPA workerは約610秒。音声読み込み・3回反復・照合・検査を含む。
成果物は約417 MiB。単発推論速度や録音時のend-to-end latencyの計測には使用しない。
全件用workerのpilot回帰検査は`validation-worker-regression-20261007-v1`へ保持した。
再現手順は[README](README.md#全validationのscore生成と閾値校正)を参照する。

次はtestの登録・query・trial metadata、実装・テスト・モデル・全音声checksum、
このvalidationのprofile・score・閾値、testを含むresource planを実行凍結する。
凍結後に固定閾値でtestを実行し、既定の全条件・10,000回speaker bootstrap・paired差を報告する。
Phase 3/6のtestは既に観測済みなので、新たな独立holdoutと呼ばない。
