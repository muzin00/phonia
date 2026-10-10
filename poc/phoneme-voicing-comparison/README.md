# 有声・無声子音の比較

母音＋`m/n`の改善が鼻音固有の寄与なのか、他の子音でも生じるのかを調べる。
JVSの既存train 70話者とCommon Voice日本語の既存train 70話者を使用し、SRC4VCは含めない。
同じ14音素モデル `a i u e o m n t d k g s z h` で、次の入力を比較する計画。

| 主比較 | 音素数 |
| --- | --- |
| 母音のみ | 5 |
| 母音＋m/n | 7 |
| 母音＋t/k | 7 |
| 母音＋d/g | 7 |

登録・照合に使う実際の音声時間を揃える。単音素の`t/d`・`k/g`も比較し、`s/z/h`は補助診断に使う。
モデル、発話文脈、音素ラベルの影響があるため、この比較だけで有声性の因果効果を断定しない。
`p/b`はCommon Voiceの一部話者で区間が不足するため、今回は含めない。

## 学習前の試聴

追加する`t d k g z h`を、コーパス・音素ごとに5区間、合計60区間用意する。
毎回100件を回答する正式レビューではなく、音素と切り出し位置を少数例で確認する。
**回答、品質OKの記録、現在位置は保存しない。** サンプルと自動チェックの記録は再現用に保存する。
試聴確認が終わるまでは学習を開始しない。

既存のtrainアライメントと元WAVから、30〜250msの境界内区間を準備する。
250msを超える区間のみ中央を切り出し、RMS −50dBFS未満を除外する。
波形のパディング・増幅・文脈追加は行わず、PCM形式、境界、元WAVのSHA-256、メタデータ、話者ごとの残存数を確認する。
レビューは学習候補の境界内区間を示す。今後の学習で長い区間のランダムcropを使う場合、その位置はこの中央cropと異なる。
短い破裂音も区間全体を残し、従来の単音素比較で使用した中央50msへの統一は適用しない。
音素ラベルは音声内の有声・無声を保証しないため、試聴では音素の位置と聞こえ方を確認する。

サンプルは自動チェック後の全候補から固定seedのSHA-256順で選ぶ。選んだ区間を聞きやすい例に差し替えない。
原区間・除外候補・残存候補・抽出サンプルを別manifestとして保持する。
前後100msと元発話は位置確認の補助で、学習波形には追加しない。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/prepare_review.py
```

出力先は`artifacts/phoneme-voicing-comparison/listening-60-20261010-v1/`。
同じ出力先を再生成して上書きしない。設定は[`config/review.json`](config/review.json)にまとめる。

```sh
cd tools/phoneme-review-ui
PHONIA_REVIEW_DATASET=../../artifacts/phoneme-voicing-comparison/listening-60-20261010-v1/review-dataset.json \
PHONIA_REVIEW_MEDIA_ROOT=../../artifacts/phoneme-voicing-comparison/listening-60-20261010-v1/media \
PHONIA_REVIEW_MODE=listen pnpm dev --host 127.0.0.1 --port 5173 --strictPort
```

`http://localhost:5173/?mode=listen`を開く。
「試聴区間」でコーパス・音素を選択し、「候補Aだけ再生」で境界内の音を聴く。
必要に応じて前後100msや原音声を再生し、「次へ」で進む。全60件の聴取や回答保存は必須にしない。

## 学習と評価

1条件・1 seed・30,000更新で、初期値から14音素の共通モデルを学習する。
encoderは従来と同じstatistics pooling＋MLP（65,920 parameters、128次元）。
1更新は10話者×5音素×2区間＝100区間で、14更新ごとに各音素を5回提示する。
全体では3,000,000区間提示、音素ごとのmicrobatchは10,714〜10,715回となる。
損失・optimizer・特徴処理は従来と同じ。最後の重みを採用し、validation診断での重み選択は行わない。

既存8音素の621,472区間へ、試聴と同じ抽出条件の追加6音素151,413区間を全件加える。
合計772,885区間で、140話者×14音素の各組に最低9区間がある。
元8音素のtrain統計と追加6音素のtrain統計を母集団モーメントで結合する。
新規6音素は最大250msの中央crop後にQCした固定波形、元の母音・m/nの長い区間は従来のランダムcrop。
音素ごとの原データ量は等しくない。140話者はラベル数で、コーパス間の実在人物の重複は未確認。

主比較は5母音＋m/n/t/d/k/gの全11音素が利用できるcommon11発話で揃える。
登録最大3秒・照合最大1秒、元音素境界内30〜250msで、全4条件の実使用フレーム数を一致させる。
時間調整で`t/d/k/g`を短くする場合は候補の終端に合わせ、それ以外は中央cropとする。
調整後の波形もRMS−50dBFS以上であることを確認する。
登録は全4条件が合格する最大の共通時間を5ms刻みで選び、照合はどれかが不適合なら発話を全条件共通で除外する。
無音pad・区間の重複使用・文脈追加は行わない。

初回の入力準備では、登録を3秒に合わせた母音の30ms切片で低音量が検出された。
モデル推論・閾値作成前に上記の登録時間選択へ修正し、評価入力をv2として再固定した。
v1の入力と変更前ソースはartifactsの`abandoned-before-inference/`に保持する。
学習データ・学習設定は変更していない。

単音素は`t/d`、`k/g`、補助的な`s/z`・`s/h`を別々の発話集合で比較する。
各ペアで登録は最大10組の同数・同じ長さの区間、照合は各音素1区間を短い方の長さに揃える。
固定seedで候補を並べ、調整後のRMSが不適合な組は両側を除外して次の候補組を使用する。
区間数が同じでも発話内位置や周辺音素を同一にした比較ではない。
全ペアでvalidation/test各15話者を含み、同じ10,000回の話者bootstrapを共有する。
t/d・k/gの通常文EER差だけは、2仮説のBonferroni補正として97.5% CIを推論前に指定する。
それ以外のCIは探索的な95% CIで多重比較補正なし。

| 固定した発話集合 | validation 通常／別テキスト | test 通常／別テキスト |
| --- | ---: | ---: |
| 主比較common11 | 290／97 | 283／107 |
| 単音素t/d | 566／276 | 559／273 |
| 単音素k/g | 480／260 | 480／264 |
| 単音素s/z | 267／60 | 266／78 |
| 単音素s/h | 290／132 | 286／132 |

各条件の閾値はvalidation通常文のみで固定し、testと別テキストには再校正せず適用する。
入力・モデル・コード・閾値のchecksumを検査し、切片の元境界、train音声との重複、反復推論一致を確認する。
別実装の監査でスコア統合、ROC/EER、FAR/FRR、bootstrapの数値も確認する。
JVS testは過去にも観測済みで、独立holdoutではない。結果は1 seed・対象発話集合に限定する。
音素ラベルは実際の有声性の正解ではなく、この比較だけで生体依存の因果効果は確定しない。

```sh
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-voicing-comparison/tests -v
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/train.py prepare
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/train.py freeze
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/train.py train
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/study.py inputs
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/diagnostics.py inputs
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-voicing-comparison/audit.py inputs
```

`study.py`と`diagnostics.py`でそれぞれ`prepare` → `validation` → `test`を実行し、
`audit.py metrics` → `report.py`で検査・集計する。
設定は[`training.json`](config/training.json)と[`evaluation.json`](config/evaluation.json)に固定する。
学習出力は`artifacts/phoneme-voicing-comparison/training-14phones-20261010-v1/`、
主比較と単音素比較は同ディレクトリ配下の`evaluation-matched-20261010-v2/`と`diagnostics-matched-20261010-v2/`。
既存のrunは上書きしない。

結果は[学習結果](training-results.md)、[HTML比較表](evaluation-results.html)、[結果の解釈](interpretation.md)で確認する。
