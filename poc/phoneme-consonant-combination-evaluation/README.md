# 子音の組み合わせ m+n・m+s・n+s の比較

同じ新規学習済み8音素モデルで、登録・照合に使う音素を変える。
母音5種を共通とし、同じ7音素の子音組み合わせを比較する。

| 条件 | 入力音素 | 種類数 |
| --- | --- | ---: |
| vowels5 | a i u e o | 5 |
| vowels_mn | a i u e o m n | 7 |
| vowels_ms | a i u e o m s | 7 |
| vowels_ns | a i u e o n s | 7 |

モデルは[JVS・Common Voice各70話者、30,000更新、1 seed](../phoneme-consonant-combination-training/README.md)で固定する。
既存7音素モデルのスコアを再利用せず、今回の4条件をすべて再推論する。
前回との直接差ではモデルと対象発話の変更が混ざるため、今回の子音追加効果とは扱わない。

## 比較条件

- 既存の登録候補WAVとvalidation/test話者分割を使用し、train話者は含めない。
- 母音は従来の各10区間、m/n/sは同じ登録候補WAVから各10区間を固定seedで選ぶ。
- 登録最大3秒・照合最大1秒。各登録・各発話の実使用PCMフレーム数を4条件で完全一致させる。
- 区間は30〜250ms、元音素境界内のcenter crop。無音pad・同じ区間の重複使用・文脈追加なし。
- /s/は学習候補と同じく最大250msのcenter crop後にRMS−50dBFS以上を確認する。
- m/nは従来の元音素区間のRMS判定を再利用し、選択時に最大250msへcenter cropする。
- 全8音素が存在し、4条件すべてが同じ時間を使える発話（common8）を推論前に固定する。
- 音素内の区間cosineを平均し、音素間は等重みで平均する。統合係数の学習・調整は行わない。
- validation通常文で各条件のFAR 1%・0.1%・EER動作点の閾値を固定し、testと別テキストに適用する。
- 全6組を事前に固定し、15話者の同じ10,000 bootstrap drawで指標差の95% CIを求める。

推論前に固定した発話数は以下のとおり。全発話種で15話者を含む。

| split | 通常文 | 別テキスト | 固定前の通常文／別テキスト |
| --- | ---: | ---: | --- |
| validation | 495 | 217 | 750／450 |
| test | 494 | 220 | 750／450 |

音素不足の発話は4条件共通で除外し、理由とコーパス内の対象率を入力JSONに残す。
表のFAR/FRR/EERはこのcommon8内の指標で、音素不足を含む全発話での性能ではない。
全条件が同じ総時間を使っても、音素ごとの時間・区間数・選択PCM・母音の統合比率は変わる。
音素数だけの因果効果を確定する実験ではない。

今回のJVS testは過去の実験でも観測済みで、独立holdoutではない。
1 seed・15 test話者・多重比較補正なしの探索的CIという範囲で解釈する。

## 実行

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-consonant-combination-evaluation/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-evaluation/study.py inputs
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-evaluation/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-evaluation/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-evaluation/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-consonant-combination-evaluation/report.py
```

`inputs`はモデルの学習完了前にも実行できる。`prepare`以降は固定した30,000更新の学習完了後に実行する。
使用済み出力先への再抽出とdesign-freezeの再作成は拒否する。
設定は[protocol.json](config/protocol.json)、出力は
`artifacts/phoneme-consonant-combination-evaluation/matched-20261010-v1/`。

入力・コード・encoderのchecksum、実使用時間、元音素境界との一致、登録／照合の音声分離、
全試行の話者・条件・正解ラベル、反復推論での出力一致を検査する。
validation閾値とbootstrap drawをtest推論前に固定し、変更があればtest処理を拒否する。

2026-10-10に学習・validation・test・全6組の比較まで完了した。
[HTML比較表](evaluation-results.html)・[結果の解釈](interpretation.md)・
[全48セルCSV](evaluation-cells.csv)・[閾値・件数・CIのJSON](evaluation-results.json)を参照する。
artifacts内の`input-independent-audit.json`と`independent-numerical-audit.json`で、
推論前の入力と実測値を別の処理でも検証している。
