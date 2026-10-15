# Common Voiceを学習から除く比較

Common Voiceの学習データがモデルの性能に与える影響を、CVあり／なしの2条件で調べる。
既存の実装・入力・完了済みモデルを変更せず、新しいrunへ保存する。RLは使わない。

## 事前に固定する条件

| 条件 | 音響学習の話者ラベル数 | 音素区間数 |
| --- | ---: | ---: |
| with_cv | 490 | 1,930,593 |
| without_cv | 350 | 1,264,413 |

without_cvはCommon Voiceの140ラベル・666,180区間を除き、JVS 70・JTube 210・SRC4VC 70ラベルを使う。
非学習128次元特徴だけを再利用し、音響encoder・分類head・Transformerはすべて未学習から始める。
CVは音響学習だけでなく、Transformerの登録音声・query・負例からも除く。
元区間への対応を保存し、すべての学習用参照について話者・音素・5母音条件を検査する。
Transformerの利用可能ラベル数は5母音の条件によって音響学習のラベル数と異なる。

学生初期値は両条件とも20261101・20261102・20261103。各モデルで音響encoderを216,000更新、
Transformerを64,000更新する。音響学習は前回の通常抽選と同じ音素・話者均等の循環候補方式、
AAMSoftmax + 0.5 SupCon、singletonの分類補助lossを使う。
正例ペアのある音素を均等に巡回する。CV除外後の`dy`は4区間すべてが別話者のsingletonとなるため、
ペア抽選は35音素、残る`dy`は分類補助lossだけで使用する。CVありは36音素（`dy`は14区間、ペア可能な2話者）。
Transformer・評価の36音素IDは両条件で同一。既存の学習器を改変せず、新しい通常抽選ループで扱う。
残った区間を同じ更新数で使うので、区間あたりの露出回数や分類headのサイズは変わる。
実際の区間投入数・ユニーク区間数・CPU時間も記録する。

Transformerは話者を均等抽選し、その話者のqueryと同一コーパスの別話者を均等抽選する。
この負例ルールは両条件で同じで、初期値ごとに全scheduleを学習前に固定する。
前回のCV専用hard-negative scheduleは再利用しない。学習済みencoderによる負例選択も行わない。
そのため、過去の結果ではなく、今回再学習するwith_cvが主対照となる。
本人・他人を同数にし、ペアの共通音素maskを使う。
学習率は初めの4,000更新が3e-4、その後3e-5。その他のモデル・optimizer設定は前回を継承する。

## 評価

JVS・Common Voice・JTubeのselection33話者とtestを両条件で固定する。
学習からCVを除いても、モデル選択と最終評価ではCVを残す。
これは「CV音声で勾配更新しない」比較であり、CVのselection情報も一切使わない実験ではない。
旧reward32話者は今回使用しない。queryとclaim両側の分割、話者・音源hashの分離を継承する。

音響encoderは固定216,000更新。Transformerは0更新と2,000更新ごとのselection macro EERを測り、
最小値の最初のcheckpointを選ぶ。全6モデルとselection FAR1%閾値を固定してからtestを採点する。
主指標は3コーパス等重み・3初期値平均のtest EER差 `without_cv − with_cv`。
各コーパスEER、初期値別結果、FAR/FRR、採点率と1,000回の対応付き話者・初期値bootstrap区間も報告する。
EERは採点可能な試行、FAR/FRRは未採点を拒否として含める。既存モデルの自動置換は行わない。

全コーパスで改善すれば除外を支持する。JVS/JTubeだけ改善してCVが悪化する場合はトレードオフとして報告する。
データ量・話者数・コーパス比率も同時に変わるため、改善しても録音品質だけの因果効果とは断定しない。
testは既観測であり、独立した未観測データへの汎化はこの比較だけでは示せない。

## 実行

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-common-voice-ablation/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-common-voice-ablation
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-common-voice-ablation/ablation.py prepare
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-common-voice-ablation/ablation.py execute --smoke
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-common-voice-ablation/ablation.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-common-voice-ablation/ablation.py status
```

smokeは別runで両条件1初期値、音響400更新＋Transformer40更新。testは採点しない。
最初のsmokeは`dy`の正例ペア欠落を検出して停止した。対応後は`-smoke-v2`を使用する。
本比較への自動縮小はしない。3プロセスまで並行実行し、checkpointから同じコマンドで再開できる。
`state.json`、条件・初期値別ログ、失敗・完了マーカーを保存する。
完了時は`results.json`と`complete.json`を保存し、設定・コード・入力のhashを再確認する。
