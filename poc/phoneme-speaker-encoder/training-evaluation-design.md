# 学習・評価設計

## 1. 目的

入力方式やencoderを公平に比較し、学習に含まれない話者の本人照合性能から方式を選択する。
validationをモデル選択に使用し、testは全設定を固定した後の最終評価だけに使用する。

## 2. 候補を比較する学習条件

各候補設定では入力、encoder、sampling、損失など複数の候補軸を組み合わせる。候補軸以外の
差を抑えるため、比較時は次を共通化する。

- train cohortと使用する区間ID
- optimizer、学習率schedule、更新回数
- checkpoint評価の間隔とearly stopping規則
- validationの登録区間、照合区間、trial一覧
- seedの集合

候補軸、組み合わせ、採用規則は[候補比較計画](comparison-plan.md)を正本とする。

まず10話者cohortで損失が低下し、登録・照合処理まで実行できることを確認する。入力方式の
選択は70話者cohortで行う。互換性のある候補の組み合わせを3 seedで学習し、単一の初期値や
単一項目だけの比較で判断しない。

front-endの違いにより完全に同一条件にできない場合は、差と理由を実験結果へ記録する。

## 3. encoderと学習目的

### 3.1 参照encoder

最初の学習成立性確認に使う参照候補として、5母音で共有する小型TDNNを定義する。最終的な
encoderは、統計pooling + MLP、TDNN、および生波形用encoderを同じ選択規則で比較して決める。

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

各TDNN blockの直後に無効時刻を0へ戻す。LayerNormは各時刻のchannel方向だけに適用し、
時間方向やbatch方向の統計へpaddingを混ぜない。最短の2 frameでもpoolingが成立するよう、
標準偏差は母分散と`1e-5`の下限から計算する。

母音ラベルはencoderへの入力にせず、samplingとcontrastive lossの組を作るためだけに使う。
照合も同じ母音のprofileとの間で行うため、母音embeddingによる追加条件付けは候補に含めない。
母音別profileはencoderの外で管理する。

### 3.2 参照batch sampling

1 batchを`P=10`話者、5母音、各話者・母音につき`K=2`区間の100区間で構成する。

- 話者はcohortから一様に重複なしで選ぶ。
- 各話者について5母音を必ず含める。
- 話者・母音内では区間を一様抽出し、同じbatch内で区間IDを重複させない。
- すべての話者がほぼ同じ回数選ばれるよう、seed付きの話者順を巡回する。
- 区間長bucketingは、この`P × 5 × K`構成を崩さない範囲だけで行う。
- trainの`evaluation_role=training`以外はsamplerへ渡さない。

これにより各anchorは、同じ話者・同じ母音のpositiveを1件、同じ母音・異なる話者の
negativeを18件持つ。区間数の多い話者や母音が更新回数を支配しない。

samplingは`P=10, K=2`に固定する。`P=5, K=4`より同母音の異話者negativeを多く確保でき、
各anchorに必要なpositiveも1件存在するためである。

### 3.3 損失関数

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
安定した話者分類目的を共通にした上でmetric lossを追加する価値を比較する。併用候補では
上記の重みを使用する。

### 3.4 共通するoptimizerと更新条件

| 項目 | 成立性確認 | 本学習 |
| --- | ---: | ---: |
| cohort | 10話者 | 70話者 |
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

seedは`20260926`、`20260927`、`20260928`の3個を本比較に使う。データ順、crop、モデル初期化、
workerのseedをrun seedから決定論的に導出する。再現性を優先する比較runでは、利用する
frameworkのdeterministic modeを有効にし、非決定的な演算を検出したら失敗させる。

## 4. 学習後の登録・照合評価

### 4.1 encoderの固定

評価するcheckpointを読み込み、encoderと入力正規化を固定する。validationの音声を使った
encoderの更新や正規化統計の再計算は行わない。

### 4.2 話者プロファイルの作成

validationの`enrollment`から、話者別・母音別に登録区間を取得する。初期の主条件は
1話者・1母音あたり10区間とする。区間IDと抽出seedを固定し、すべてのモデルで同じ区間を
使用する。

各区間から得たembeddingをL2正規化し、その平均を再度L2正規化して話者・母音別の
登録プロファイルとする。登録時にencoderは再学習しない。

登録区間数による変化は、1、5、10区間を補助条件として評価する。

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

FARとFRRは別々に報告する。genuineとimpostorの試行数が異なるため、両者の誤り件数を
単純に合算したaccuracyを主指標にしない。

モデル間の差は同じ話者とtrialを使った対応のある比較とし、必要に応じて話者単位の
bootstrapで区間を推定する。validation話者が15人であるため、小さな差を確定的な優劣と
解釈しない。差が小さい場合は、再現性と計算コストを含めて単純な方式を優先する。

## 5. checkpointと閾値の選択

各学習runでは、決められた間隔でvalidationの登録・照合評価を行う。主条件のmacro EERが
最小のcheckpointを候補とし、同値または差が小さい場合はcross-text、短区間性能、計算量を
補助判断に使用する。

実運用を想定した判定閾値はvalidationだけで選択する。初期PoCでは、EER閾値に加え、
目標FARを固定したときの閾値とFRRを記録する。PoCの主条件は目標FAR 1%とし、0.1%も
補助値として記録する。いずれもvalidation impostor scoreから閾値を固定し、その閾値を
testへそのまま適用する。

## 6. testによる最終評価

入力方式、前処理、encoder、checkpoint選択規則、登録区間数、スコア、閾値をvalidationで
確定した後、testの15話者で同じ登録・照合手順を一度実行する。

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
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── selections/
│   ├── enrollment-segments.jsonl
│   └── trials.jsonl
├── scores/
│   ├── validation.jsonl
│   └── test.jsonl
└── metrics/
    ├── validation.json
    └── test.json
```

experiment IDは`<UTC YYYYMMDDTHHMMSSZ>-<git short sha>-s<seed>-<config sha先頭8桁>`とする。
`config.json`はすべての確定パラメータを展開したJSON、`environment.json`はOS、Python、
framework、CUDA、device情報、`data.json`はmanifestとtrain特徴統計のpath・SHA-256を持つ。

checkpointにはmodel、optimizer、scheduler、update、run seed、config SHA-256、manifest
SHA-256、特徴統計SHA-256を含める。不一致のcheckpointは明示的なoverrideなしに読み込まない。
score JSONLはtrial ID、split、role、speaker、claimed speaker、母音、区間ID、区間長、登録区間数、
cosine scoreを必須とする。test成果物は最終設定を固定するまで作成しない。

## 8. 小規模な学習成立性確認

本比較へ進む前に次を順に満たす。

1. 固定した32区間を過学習させ、500 update以内にtrain話者分類accuracy 95%以上となる。
2. 10話者cohortを2,000 update学習し、損失、gradient、embeddingが有限値を保つ。
3. enrollment profile、genuine/impostor trial、macro EERまでを一つのrun IDで生成できる。
4. 同じcheckpointとtrialで評価を2回実行し、scoreと指標が一致する。
5. 異なる長さを混ぜたbatchでpadding不変性テストが通る。

成立性確認のEER自体は方式選択に使わない。失敗した場合はtestを実行せず、原因と変更した
設定を新しいrunとして記録する。

## 9. 後続Issueへの分割

実装は依存順に次のIssueへ分けられる。

1. **Dataset / DataLoaderと特徴統計**: source slice読出し、入力変換、mask、balanced sampler、
   train統計、enrollment/trial固定、単体テスト。
2. **encoderと学習**: TDNN、pooling、AAM-Softmax、within-vowel SupCon、checkpoint、
   10話者成立性確認。
3. **登録・照合評価**: profile、cosine score、EER/FAR/FRR、区間長別集計、曲線、固定閾値。
4. **70話者比較実験**: 候補の組み合わせ、3 seed、checkpoint選択、validationによる方式確定。
5. **test最終評価**: 選択済み設定を凍結し、testを一度だけ評価して結果を文書化。

各Issueは前段の成果物schemaを変更しない。変更が必要な場合はschema versionを上げ、既存runと
混在させない。

## 10. 設計上の保留事項

- 生波形encoderの具体構造は、探索空間manifestを生成する前に比較候補として固定する。
- 実運用で許容するFARは製品要件が定まった時点で決める。PoCでは1%を主条件とする。
- 70話者比較で計算資源が不足する場合はbatchの`P`を変えず、gradient accumulationまたは
  precisionを変更し、実効batchの構成を維持する。
