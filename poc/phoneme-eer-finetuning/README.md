# EERに近い損失によるTransformer追加学習

本人／他人のBCEで学習しvalidation EERで選ぶ現在の方式に対して、
重み更新にもEERの近似を加える効果を検証する。強化学習は使わない。
元のSRC490・36音素モデルを維持し、別runで対照と介入を比較する。

## 事前固定した比較

- 最新SRC490の3音響seed。encoderは490ラベル、統合層の入力は485ラベル。
- 各seedの既存64k予算内のvalidation選定Transformerを出発点にする。
  34音素除外・再学習や隣接音素モデルは使わず、従来の36音素を保持する。
- 追加前、BCE＋従来KLの追加学習、BCE＋従来KL＋soft-EERの3条件。
  両学習条件は同一の出発重み、AdamW新規初期化、dropout乱数を使う。
- 元の学習ペア列の56,000〜63,999行を再利用し、連続4行を1batchにまとめる。
  1更新128 query・本人128／他人128の256照合、2,000更新。
  CV限定難例を含む既存の負例を保持し、再選定しない。
- 両条件とも学習率1e-5、weight decay 0.01、KL係数0.001、clip 5。
  soft-EERの係数は1、温度は0.03に固定。温度・係数の探索はしない。
- 追加0と250更新ごとに、JVS・CV・JTubeの通常文validation EERを測る。
  3コーパス等重み平均の最小、同値は早いcheckpointを採用する。
  条件ごとの選定モデルと、固定2,000更新終点を両方報告する。
- 全モデルと各validation FAR1%閾値を確定してからtestを推論する。
  EER、全入力FAR/FRR、採点率、音響seed別対応差、JVS別文を報告する。
  対応付きquery/claimed話者bootstrapは1,000回。

出発点は学習済みモデルだがoptimizerを揃えてリセットし、batchも大きくする。
従来64kの厳密な継続ではなく、同一条件の追加学習で損失だけを比較する実験である。
学習ペア数は各条件512,000照合。音声の追加、encoder更新、評価入力の変更はない。

## soft-EERの定義

本人の類似度をs+、他人をs−、閾値をt、温度をTとする。

```
soft-FAR(t) = mean(sigmoid((s− − t) / T))
soft-FRR(t) = mean(sigmoid((t − s+) / T))
soft-FAR(t*) = soft-FRR(t*)
soft-EER = (soft-FAR(t*) + soft-FRR(t*)) / 2
loss = BCE + 0.001 KL + 1.0 soft-EER
```

閾値はfloat64・60回の二分探索で解く。等誤り条件の陰関数微分を使い、
閾値自身がスコアに依存する分もbackwardに含める。
閾値を固定したまま微分する方法や、二分探索の分岐をそのまま自動微分する方法ではない。
sigmoidの微分はlog空間で安定化し、FAR/FRRの差を検査する。
近似には校正前の加重類似度を用い、学習可能なscale/biasで近似を操作させない。
閾値は学習batch内だけで求め、運用閾値として使わない。

近似の単位は0〜1の率。各batchの全コーパスをプールするので、
コーパスごとに全件測る実EERと完全に同じ目的ではない。
現行の各音素類似度の加重平均というモデル構造も保持している。

関連研究として、[EERLoss](https://arxiv.org/abs/2606.24586)はEERの近似を学習に利用し、
[Mingote et al.](https://www.isca-archive.org/interspeech_2019/mingote19_interspeech.html)は
話者照合のFAR/FRRを滑らかに近似したDCF損失を検討している。
今回はsigmoidと陰関数微分による独自の小規模比較であり、いずれかの論文の再現実験ではない。

## 検証と実行

本学習前に、数値微分との一致、スコア同値・分離・逆転、共通シフトへの不変性、
係数0での元BCE損失／勾配との一致、学習再開の完全一致を検査する。
モデル・optimizer・乱数・best・履歴を同じatomic checkpointに保存する。
同じexecuteで再開できる。既存成果物と入力はhashで固定し、終了後にも確認する。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-eer-finetuning/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-eer-finetuning
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-eer-finetuning

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-eer-finetuning/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-eer-finetuning/study.py status
```

3 CPU worker、各1 threadで、各workerがBCEとsoft-EERを順に学習する。
条件は[config.json](config.json)。成果物はそのrun_directoryに保存する。
モデル推論を直接forwardでも再計算し、旧モデルのスコア一致、EER別実装、
閾値、未採点位置、選択規則、乱数の一致を監査する。

全testは以前の実験で参照済みで、新しい話者集団での確認ではない。
3音響seed・1追加学習seed、単一温度・係数・予算の比較であり、
改善しなくてもEERを目標とする学習全般を否定しない。
EERは採点可能分、FAR/FRRは未採点を拒否として含む。
validationで閾値を固定するためtest FARは条件間で完全には揃わない。
標準モデルや認証閾値の置換はこの実験に含めない。

完了後の[結果表](evaluation-results.md)・[全条件と監査](evaluation-results.json)。
