# 音素ごとの識別力を直接比較する

「日本語の音素によって識別力が異なり、mとnはsより有効」という仮説を、
同じ学習済みencoderで取り出せる識別情報として検証する。
母音5種＋m/n/sで学習済みの固定8音素モデルを使用し、追加学習は行わない。

## 固定する条件

| 項目 | 条件 |
| --- | --- |
| 比較音素 | a i u e o m n sをそれぞれ単独使用 |
| 登録 | 各音素10区間×50ms＝0.5秒 |
| 照合 | 各音素1区間×50ms |
| 抽出・品質判定 | 全音素で元境界内のcenter crop後、RMS−50dBFS以上 |
| 区間選択 | 固定seed、登録は元WAVを分散。embeddingやスコアを使わない |
| 話者・発話 | 既存common8から全音素の50ms区間が使える発話を推論前に固定 |
| 閾値 | validation通常文で音素別に固定し、test・別テキストに適用 |
| 主検証 | 通常文testのm−s、n−sのEER差 |
| 差の区間 | 15話者、同じ10,000 bootstrap draw。主2比較は各97.5% CI |
| 補助検証 | 全28組の95% CI、FAR/FRR、別テキスト、登録話者別EER |

50msはスコアを見る前に区間の長さと利用可能件数だけで決めた。
既存母音候補の80〜100ms条件では/i/・/u/等の不足により共通発話や登録話者が大きく減るため、
全8音素・15話者を揃えられる50msを単一条件として採用する。
登録は従来と同じ候補WAVから全8音素のraw境界を再抽出し、10区間ずつ使用する。

## 測定するもの

- 単独音素のEERと、validation目標FAR 1%・0.1%の閾値でのFAR/FRR。
- 本人/他人のcosineスコア分布、平均・標準偏差、平均差、標準化した分離度d′。
- 単一区間の正規化embeddingの話者間分散・話者内分散とその比。話者を等重みにする。
- 登録話者別EERとスコア分布。各話者で音素の寄与が変わるかを診断する。

主仮説はm−s、n−sの両EER差の97.5%区間上端が0未満の場合に支持とする。
Bonferroniで主2比較を扱い、その他の比較と話者別順位は探索的診断とする。
端点が0の区間は0を含むものとして扱う。

この比較は「今回のモデルが取り出せた情報」を測る。
音素本来の識別情報、発声・調音・声道形状のどれが原因かを確定するには、別モデル・音響特徴や文脈を揃えた追加検証が必要。
50ms・全8音素が揃う発話・観測済みJVS test・1 seedの結果であり、未知録音条件・独立holdoutでの確証ではない。
前回の登録3秒／照合1秒の複数音素比較とは入力と対象発話が異なる。

## 実行

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-discriminability/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-discriminability/study.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-discriminability/study.py validation
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-discriminability/study.py test
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-discriminability/report.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-discriminability/audit.py
```

設定は[protocol.json](config/protocol.json)。使用済み出力先での再抽出は拒否する。
trainと評価の話者・音声checksumの分離、境界・実時間・区間数、反復推論一致、encoder不変、
閾値のtest前固定を検査する。独立監査で元PCMからのRMS、スコア、率、EER、分布、分散、CIを再計算する。

[比較HTML](evaluation-results.html)・[結果の解釈](interpretation.md)・[全表](evaluation-results.md)

## 実測完了

validation通常文462・別テキスト140発話、test通常文448・別テキスト148発話、各15話者で実測した。
通常文testのEERはm＝5.134%、n＝6.091%、s＝16.295%。主比較2件の97.5% CIはともに0を含まず、
固定モデル・50ms区間でのm/nの優位を確認した。別テキストも同方向。
話者による違いと適用範囲は[結果の解釈](interpretation.md)に記載する。

8件の単体テスト、Ruff、反復推論bitwise一致、encoder不変、独立再計算監査が通過した。
runの`independent-audit.json`に143,760試行、11,984 PCM区間、96率セル、32 EERセル、480話者診断、
16 validation FAR閾値、168 paired配列と主2比較の調整区間の確認結果を保存する。

Gitに保存する生成CSV/SVGは、改行をLFに揃え、行末空白を除去する。
元出力はrunの`publication-before-git-formatting`に保存し、CSVの全フィールドとSVGのXML内容の一致を
`git-formatting-audit.json`に記録する。推論コード・設定・測定値は変更しない。
