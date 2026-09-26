# 学習・評価設計

## 1. 目的

入力方式やencoderを公平に比較し、学習に含まれない話者の本人照合性能から方式を選択する。
validationをモデル選択に使用し、testは全設定を固定した後の最終評価だけに使用する。

## 2. 入力方式を比較する学習条件

入力方式以外の差をできるだけ抑えるため、比較時は次を共通化する。

- train cohortと使用する区間ID
- 話者・母音を均衡させるbatch sampling
- 学習目的と損失関数
- embedding次元と可能な範囲のモデル規模
- optimizer、学習率schedule、更新回数
- checkpoint評価の間隔とearly stopping規則
- validationの登録区間、照合区間、trial一覧
- seedの集合

まず10話者cohortで損失が低下し、登録・照合処理まで実行できることを確認する。入力方式の
選択は70話者cohortで行う。最終候補は少なくとも3 seedで学習し、単一の初期値だけで
判断しない。

front-endの違いにより完全に同一条件にできない場合は、差と理由を実験結果へ記録する。

## 3. 学習後の登録・照合評価

### 3.1 encoderの固定

評価するcheckpointを読み込み、encoderと入力正規化を固定する。validationの音声を使った
encoderの更新や正規化統計の再計算は行わない。

### 3.2 話者プロファイルの作成

validationの`enrollment`から、話者別・母音別に登録区間を取得する。初期の主条件は
1話者・1母音あたり10区間とする。区間IDと抽出seedを固定し、すべてのモデルで同じ区間を
使用する。

各区間から得たembeddingをL2正規化し、その平均を再度L2正規化して話者・母音別の
登録プロファイルとする。登録時にencoderは再学習しない。

登録区間数による変化は、1、5、10区間を補助条件として評価する。

### 3.3 trialの作成

`verification`の各照合区間について、同じ母音の15話者分の登録プロファイルと比較する。
1区間につき、同一話者との1試行をgenuine、異なる14話者との14試行をimpostorとする。

登録と照合で同じ元音声を使用しない。trial一覧を事前に成果物として固定し、入力方式間で
共通利用する。

主要評価とは別に、`cross_text_verification`を同じ方法で照合し、登録文と異なる文章での
頑健性を確認する。JVSには収録日とセッションIDがないため、この結果をcross-session性能
とはみなさない。

### 3.4 スコア

初期スコアは、照合embeddingと同じ母音の登録プロファイル間のcosine類似度とする。
母音ごとのスコアを一つの認証スコアへ統合する方式は
[Issue #22](https://github.com/muzin00/phonia/issues/22)で扱う。

### 3.5 指標

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

## 4. checkpointと閾値の選択

各学習runでは、決められた間隔でvalidationの登録・照合評価を行う。主条件のmacro EERが
最小のcheckpointを候補とし、同値または差が小さい場合はcross-text、短区間性能、計算量を
補助判断に使用する。

実運用を想定した判定閾値はvalidationだけで選択する。初期PoCでは、EER閾値に加え、
目標FARを固定したときの閾値とFRRを記録する。目標FARの具体値は、想定する利用条件を
決めてから固定する。

## 5. testによる最終評価

入力方式、前処理、encoder、checkpoint選択規則、登録区間数、スコア、閾値をvalidationで
確定した後、testの15話者で同じ登録・照合手順を一度実行する。

testでは次の二種類を区別して報告する。

- test内で曲線から算出するEER、ROC、DET: 閾値に依存しない最終的な分離性能
- validationで固定した閾値をそのまま適用したFAR、FRR: 閾値の一般化性能

testの結果を理由に入力方式や閾値を変更した場合、その結果を最終評価として扱わない。
変更後の方式はvalidationで再選択し、別の未使用評価データを用意する。

## 6. 保存する成果物

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

## 7. 未決事項

- encoder構造とembedding次元
- 母音ラベルの条件付け方法
- batch sampling
- 学習目的と損失関数
- checkpoint評価間隔とearly stopping条件
- 目標FARと運用閾値
- 実験成果物の具体的なスキーマと配置
