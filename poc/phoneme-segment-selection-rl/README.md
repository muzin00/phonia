# 個別音素区間を選ぶ学習

音響encoderを未学習状態から学習し、その時点で使う区間を共通の小さな方策モデルで選ぶ。
既存の音素種類を重視する発話選択実験とは別の実験で、旧データ・実装・モデルは変更しない。

## 対象と選択

元の1,930,593区間・490話者・36音素の非学習128次元特徴キャッシュを使用する。
1,929,913区間は同じ話者・音素に相方があり、残り680区間は分類lossだけで使う。
singletonは全条件共通で32更新ごとに1区間、通常の分類lossに0.01倍で加える。

元encoderと同じ音素順・話者順の5音素×最大10話者×2区間を維持する。
各話者・音素内のシャッフル済み循環キューから候補を出し、最大4組から1組を抽選する。
十分な区間がある場合は8区間を重複なしで4組にする。少数群は重複しない組だけを候補にする。
稀な音素の不足枠は両lossからマスクする。組内2区間の同一録音は禁止せず、方策の入力にする。

方策入力は、2区間の128次元特徴の平均・絶対差、長さ・音量、現在のencoder/headによる
正解話者cosine・他話者との差、区間同士の類似度、使用回数、同一録音フラグ、学習進捗、音素。
話者IDと検証ラベルは方策に入力しない。32次元隠れ層のMLPと音素埋め込みを使う。
採用確率はsoftmaxと20%一様探索の混合。推論時の区間削除は行わない。

## 4条件と報酬

| 条件 | 方策の学習 |
| --- | --- |
| uniform | 通常抽選。方策なし |
| proxy | 20更新ごとに、1回のAdamW更新後の検証照合BCEの差で学ぶ |
| eer | 200更新ごとに、後段Transformerを含む実EERの差で学ぶ |
| hybrid | proxyの最初の2軌跡を共有し、その後6軌跡をEER報酬で学ぶ |

全条件でencoderは元のAAMSoftmax + 0.5 SupConを使用する。
proxyはDVRL/DDSから着想した**一更新の検証損失による近似**で、DDSの勾配内積実装ではない。
proxyの検証はreward話者の本人・他人を同一queryに対応させ、各コーパス8queryずつ使う。
訓練話者の分類正解率は検証報酬に使わない。

EER方式は同じencoder、分類head、Transformer、各optimizer状態から通常抽選と方策抽選に分岐する。
候補・更新数・dropoutは揃えるが、**通常抽選の乱数は方策の行動と独立**にする。
同じ抽選乱数を使うとREINFORCEのbaselineが行動と相関し、初期一様方策も動かなくなるため。
encoder更新後に各分岐の特徴を再計算し、Transformerを同じ訓練発話で8回更新する。
報酬は `通常抽選のmacro EER − 方策抽選のmacro EER`。報酬が負でも方策分岐を継続する。

50ブロックごとに両分岐をさらに同一の通常抽選で最大1,000更新し、Transformerも更新してEERを測る。
その回は短期報酬と長期報酬の等重み平均を使う。先読みモデルは破棄し、継続更新数を変えない。
有限の先読みであり、最後までの完全な遅延報酬や区間単体の因果的価値を求めたとは解釈しない。

## 本比較の予算

各学習軌跡は216,000更新。方策は8つの学生初期値の軌跡を経験する。
2・4・8軌跡後の方策を保存し、学習時と異なる3初期値でencoderを216,000更新ずつ新規学習する。
この評価では方策を固定し、rewardデータで追加調整しない。
その後、各encoderに対してTransformerを新規に64,000更新する。
最初の4,000更新は学習率3e-4、以降3e-5。全条件で同じ既存CV-only負例scheduleを使用する。
selection EERを2,000更新ごとに測り、最小値の最初のTransformerを選ぶ。

主比較は事前固定した8軌跡後。2・4軌跡はselection側の学習曲線として公開する。
通常抽選は3初期値、各学習方策は3予算×3初期値。hybridのproxy段階は実体を共有して重複計算しない。
CPU秒と実更新数を記録する。同じ学生更新数での比較であり、同じ総計算時間での優位性は別途検証が必要。

既存のreward32話者・selection33話者・testを継承し、queryとclaim双方で分割する。
話者と音源hashの重複、元キャッシュhashを確認する。全モデル・selection閾値を固定してからtestを採点する。
評価対象は通常文の3コーパス等重みEER、selection FAR1%閾値でのtest FAR/FRR、採点可能率。
対応付き話者・学生初期値bootstrapを使うが、方策初期値は各条件1つで、その変動は含まない。
既存testは過去に観測済み。最良モデルへの自動置換は行わない。

## 実行・再開

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poc/phoneme-verification-evaluation/.venv/bin/python \
  -m unittest discover -s poc/phoneme-segment-selection-rl/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-segment-selection-rl
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-segment-selection-rl/study.py prepare
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-segment-selection-rl/study.py execute --smoke
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-segment-selection-rl/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-segment-selection-rl/study.py status
```

`--smoke`は別runで400更新×2軌跡、Transformer40更新、学生1初期値。
データ接続・分岐・再開・集計の動作確認専用でtestを採点せず、優劣の結論を出さない。
本比較への自動縮小はしない。本比較は長時間かかる。

実装・設定・入力を固定し、同じコマンドで完了済み処理をスキップして再開する。
ブロックごとに方策・学生・全optimizer・候補キュー・区間使用回数を圧縮checkpointに保存する。
抽選とdropoutは初期値・用途・更新番号から決まり、再開時にも一致する。
最終Transformerは2,000更新ごとに保存する。実装変更時は別runが必要。

`state.json`と条件別logで進捗を確認できる。全工程完了後にrun内の`results.json`へ
学習曲線、EER/FAR/FRR、対応付き区間、区間のカバレッジ、計算量をまとめる。

背景: [DVRL](https://proceedings.mlr.press/v119/yoon20a.html)、
[DDS](https://proceedings.mlr.press/v119/wang20p.html)。今回の組選択・報酬・先読みは独自の構成。
