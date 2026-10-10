# 音素種類数を増やしたときの統合EER

既存14音素encoderを固定し、5母音を起点に5・7・10・14種類を照合に使う。
音素単独のEER順位で追加順を選ばず、種類数と統合EERの関係を測る。
学習する音素数の効果ではなく、同じモデルから照合に使う音素の種類数の検証。

## 固定条件

- 共通モデル：JVS70＋Common Voice日本語70、30,000更新、65,920 parameters、128次元。追加学習なし。
- 14音素：a i u e o m n t d k g s z h。
- hashで並べ替えた9子音と9循環シフトから、入れ子の追加順を事前固定。
- 各7種類では全子音が2回、10種類では5回ずつ現れる。各9組合せの統合EER平均と範囲を報告する。
- 5種類と14種類は各1条件、合計20条件。9順は独立標本ではなく、最良条件をtestで選ばない。
- 全14音素が使える同じ単一発話を全条件で評価し、validation/testは別々の15話者。
- 登録上限3秒・照合上限1秒。全20条件が合格する最大共通PCM時間を5ms刻みで決める。
- 境界内30〜250ms、RMS−50dBFS以上。長い候補は中央250ms、短縮したt/d/k/gは候補末尾、他は中央。
- 同一条件内で元区間の再使用・重複・pad・増幅・文脈追加を行わない。
- 同じ元音声プールを使うが、条件ごとに取り出す切片・音素内の区間数は異なる。
- 音素内cosine平均を音素等重みで統合し、validation通常文で条件別FAR1%/0.1%閾値を固定してからtestへ適用。
- 主比較：通常文の14−5種類の統合EER差、対応付き話者bootstrap95% CI、2,000回。
- 7/10種類の平均EER、隣接差、FAR/FRRは補助。CIは固定モデル・追加順・validation閾値に条件付く。
- 別テキストは対応する発話・話者が少なく、記述値だけを出す。CIによる主判定は行わない。

## 再現

設定は[protocol.json](config/protocol.json)。既存のPOC・artifactは変更しない。
使用済みrunへの推論再実行を拒否する。別runは未使用のrun directoryを設定して開始する。

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-count-scaling/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/audit.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/report.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/normalize_publication.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-count-scaling/verify_publication.py
```

`prepare`は入力と追加順・コード・モデル・元PCM・raw境界・runtimeを凍結する。
学習話者と評価話者、学習音声と評価音声、登録音声と照合音声の分離を確認する。
validationの閾値とbootstrapをtest推論前に別freezeで固定する。
反復推論のbitwise一致と重み不変を確認し、元PCM・統合スコア・EER・閾値・bootstrapを独立監査する。
Git公開時は生成SVGの行末空白とCSVのCRLFを正規化する。元ファイルと公開台帳をartifactへ保存し、SVGの内容とCSV全セルの一致を確認してから公開物を再検証する。推論コード・入力・実測値は変更しない。

## 解釈範囲

全14音素が揃う発話だけに限定され、一般的な短い発話や音素不足を含む全入力性能ではない。
同じ総時間では音素数に応じて1音素あたりの音声量も変わるため、種類数を増やす運用上の配分効果を含む。
9子音は均等に出現するが、全組合せを網羅しておらず、交互作用や追加順の不確実性までCIに含めない。
1モデル/1 seed・過去にも観測済みのJVS testであり、独立holdoutでの一般化は未確認。
前回の母音＋m/n 1.767%とは発話集合・登録候補も異なるため、その数値との直接比較ではない。

[解釈](interpretation.md)・[HTML](evaluation-results.html)・[Markdown](evaluation-results.md)・[CSV](evaluation-cells.csv)・[JSON](evaluation-results.json)
