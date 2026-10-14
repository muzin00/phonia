# EERを報酬にする音素重み・音素選択の強化学習

利用者が希望した2案を、同じSRC490学習済みTransformerから試す。
前のsoft-EER実験と異なり、実測したEERを報酬にするREINFORCEを使う。
状態は照合ペアの音素特徴、行動は重みまたは音素集合、報酬は学習batchのEER。
長期の状態遷移を持たない、一段階の方策学習（contextual bandit）である。

## 固定した条件

- SRC490の3音響seed、36音素。encoderは固定。BCE・重みRL・選択RLの計9学習。
- sourceのvalidation選定64kモデルから開始。学習話者485ラベルと既存ペアを再利用。
- 旧scheduleの56,000〜63,999行を16行ずつまとめる。512発話・本人512/他人512照合、500更新。
- 学習率1e-5、AdamW weight decay 0.01、gradient clip 5。全条件でdropoutを無効にする。
- BCE対照は従来のBCE＋0.001均等重みKL。RLはBCE/soft-EERを使わず、方策KL係数0.01。
- RLの各batchで独立4rollout。入力、音響encoder、ペア列、更新数は固定するが、計算量は同じではない。
- validationを0および100更新ごとに測定。通常文JVS/CV/JTube等重みEERの最小、同値は最初を選ぶ。
- validation選定と固定500更新を両方報告。全モデルとvalidation FAR1%閾値を固定してからtest推論。
- test話者bootstrap 1,000回。音響seed間・条件間の対応を保つ。元の5母音採点条件は全条件共通。

## 報酬と勾配

学習batchに含まれるJVS/CV/JTube/SRC4VCごとに、本人/他人スコアの実測EERを求める。
閾値を走査したFAR/FRR交点の線形補間であり、微分可能な近似損失ではない。
報酬は `r[k,c] = -EER[k,c]`。4rolloutのうち自分を除く3回の平均をbaselineにする。

```
A[k,c] = r[k,c] - mean(r[other rollouts,c])
L = -mean_over_rollout_and_corpus(A[k,c] * sum_trials_in_c(log probability(action)))
    + 0.01 * mean_trial(KL(policy || initial policy))
```

EERはbatch全体で決まるため、各コーパスの行動の同時確率（log確率の和）を使う。
報酬とbaselineをdetachし、行動も再パラメータ化せず抽選する。label/corpus/speaker IDは方策へ渡さない。
各queryの本人・他人を対にするため、報酬を計算する各コーパスに両クラスが存在する。
validation/test EERからは勾配を作らない。

## 重み付け方策

既存Transformerの36音素logitを平均とする正規分布から行動を抽選し、softmaxで重みにする。
欠けた音素は確率密度・重み・KLから除く。分散は学習しない。
探索幅sigmaは各seedの最初の**学習**batchの最大/次点logit差の中央値を[1,10]へ制限して固定する。
これは元モデルの重みが鋭いための尺度校正で、報酬や評価EERによるパラメータ探索ではない。
方策KLは同じsigmaの初期正規分布に対する厳密なKL。
推論は平均logitのsoftmaxで決定的に行い、更新0は従来モデルと一致する。

## 音素選択方策

既存Transformerのbackboneを複製し、出力headを0初期化する（各音素の採用確率0.5）。
利用可能な音素ごとのBernoulliから、少なくとも1音素を含む条件付き集合を抽選する。
logitを[-4,4]に制限。空集合は再抽選し、条件付き分布の正規化項をlog確率とKLの両方に含める。
学習済みの元Transformerを凍結したscorerが、選ばれた集合をmaskとしてattentionから再計算する。
単に従来の重みを再正規化する実装ではない。

推論は採用確率>=0.5。空になる場合は利用可能な最大確率の1音素を採用する。
更新0は全音素を採用するので従来スコアと一致。5母音は入力の採点条件として維持するが、
行動で選ばれる集合に5母音すべてを強制しない。音素数削減の報酬は含めない。
追加の方策ネットワークを持つため、重み付け方策と同容量の構造ではない。

## 検証と再実行

条件付き分布の全列挙による正規化・KL・score-function勾配、正規分布の解析勾配、
実測EERの独立実装との一致、欠損/空集合、元モデルとの一致、3条件の中断再開一致をテストする。
本学習前にtrainだけで探索幅・報酬変動・非ゼロ勾配を確認し、入力とコードをhash固定する。
元scorerの不変性、同じschedule、checkpoint選定、直接forwardとの全スコア一致、
旧スコア一致、閾値、採点対象、EER/FAR/FRRの再計算を終了時に監査する。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-eer-reinforcement/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-eer-reinforcement
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-eer-reinforcement
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-eer-reinforcement/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-eer-reinforcement/study.py status
```

3 CPU worker・各1thread。各workerがBCE、重みRL、選択RLの順に処理する。
中断後は同じexecuteで再開。モデル・AdamW・乱数・履歴・最良モデルをatomic保存する。
実験条件は[config.json](config.json)、保存先はそのrun_directory。

## 解釈上の範囲

最適化するのは「抽選した行動での学習batch EERの期待値」であり、決定的推論の全コーパスEERと同一ではない。
3音響seed・1方策seed・単一設定の試行で、手法の最良性能を探索する実験ではない。
全testは以前に参照済み。bootstrapは固定モデルの話者変動を表し、学習やモデル選択の不確実性を含まない。
EERは採点可能分、FAR/FRRは未採点を拒否として含む。標準モデルの置換は行わない。

関連する[Kanervisto et al., Odyssey 2020](https://www.isca-archive.org/odyssey_2020/kanervisto20_odyssey.html)は、
話者照合とspoof判定の結合をREINFORCEで学習した研究である。
目的はt-DCF系の報酬で、今回の音素選択・EER報酬を検証した論文ではない。
score-function推定の背景は[Monte Carlo Gradient Estimation in Machine Learning](https://www.jmlr.org/papers/v21/19-346.html)も参照する。
