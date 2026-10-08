# Phase 8: 学習データを増やす

比較条件は **現行JVS 70話者／JVS＋Common Voice日本語70話者の2条件**。
現行seed 20260926の学習結果を基準として再利用し、追加条件を同じseedで1 runだけ学習する。
encoderは既存の65,920 parameters・128次元`statistics_mlp`で固定する。

追加学習は2026-10-08に完了した。70話者ラベル・8,283発話から198,446区間を追加し、学習区間は計522,359件。
18,000 updateで早期終了し、15,000 updateのcheckpointを採用した。
母音単一区間のvalidation macro EERは現行9.196%、追加条件9.274%で、今回のvalidationでは改善を確認できなかった。
学習後の重み・特徴統計を保存し、再読込した128次元embeddingの8区間bit一致を確認した。
[学習結果](training-results.md)、[HTML比較表](training-results.html)、[全数値CSV](training-comparison.csv)を参照する。

`training-results.*`は学習完了時点の記録。発話単位の認証評価は別工程として、
[評価結果](evaluation-results.md)・[HTML比較表](evaluation-results.html)に保存する。

学習設定の正本は[protocol.json](config/protocol.json)。学習率・損失・batch・前処理・最大30,000 update・
1,000 updateごとのvalidation・early stoppingはPhase 3を引き継ぐ。
追加条件は事前学習済みモデルへの継ぎ足しではなく、同じseedの初期化から学習する。
話者分類headは140ラベルへ増えるが、推論に配布するencoderの規模は変わらない。

## 追加データ

[Common Voice 17日本語の話者別派生データ](https://huggingface.co/datasets/masuidrive/cv-corpus-17.0-ja-client_id-grouped)
のrevision `54e34c20140d041958ef0cbc1f75930eba3f55d8`、**trainだけ**を使う。
配布元はCC0として公開し、client_idを話者ラベルとしている。
80発話以上あるラベルを発話数の降順、同数なら固定seed hashで並べ、先頭70ラベルを選ぶ。
各ラベル最大130発話を固定hashで選び、既存と同じJulius・G2P・30 ms以上・near_silent除外で母音を抽出する。
データや話者の選定に認証指標を使わない。機械的な失敗・除外は全件記録する。

元コーパスのvalidationはtrainと同じclient_idを含むので、この実験の未知話者validationに使わない。
JVSのtrain/validation/test分割・既存validation trialを保持し、testは学習・選択に使用しない。
client_idは匿名ラベルであり、JVS評価話者と実在人物が重複しないことを保証できない。
人物の再特定は行わない。話者ラベル数・音素区間数・収録環境の多様性が同時に増える比較であり、
純粋なデータ量だけのスケーリング則や、独立holdout性能を示すものではない。

特徴統計はtrainだけで計算する。凍結済みJVS train統計と追加trainの統計を、
frame数と平均の差を含む母分散の結合式で統合する。validation/testの音声を統計に入れない。
正本のJVS manifest・Phase 3学習コード・本番encoder選択は変更しない。

## 再現

追加データの準備には、データ準備用Python 3.12環境の`pyarrow==21.0.0`と既存`pyopenjtalk`、
固定済みJulius、macOSの`afconvert`が必要。学習には既存Phase 3のPyTorch環境を使う。
設定は1つのprotocolに固定し、CLIでは工程だけを選ぶ。

```sh
uv pip install --python poc/phoneme-alignment-evaluation/.venv/bin/python pyarrow==21.0.0
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-training-data-expansion/prepare_data.py download
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-training-data-expansion/prepare_data.py prepare
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-training-data-expansion/training.py statistics
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-training-data-expansion/training.py train
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-training-data-expansion/build_report.py
```

成果物はGit管理外の`artifacts/phoneme-training-data-expansion/training-data-20261008-v1/`へ保存する。
配布ファイルと元WAVのchecksum、選択、失敗、区間manifest、特徴統計、学習前freeze、
学習履歴・optimizer境界checkpoint・validation指標を保持する。
中断後は同じコマンドで、checksumの一致したデータと最後のcheckpointから再開できる。
`expanded/bundle/encoder.pt`は学習headとoptimizerを含まない推論用の重み。
完了時に`expanded/training/summary.json`へstatus、採用update、validation EERと成果物hashを記録する。
`build_report.py`は完了した2条件の結果を、Markdown・HTML・CSV・JSONとしてこのディレクトリへ公開する。

## 発話単位評価

2026-10-08に評価まで完了した。validation目標FAR 1%の閾値で、testの全入力FRRは通常文4.533→3.067%、
別テキスト14.667→13.556%へ下がった。一方、全入力FARは通常文0.714→2.629%、別テキスト0.730→1.905%へ上がった。
発話単位EERも通常文1.526→1.769%、別テキスト1.257→1.330%となった。
FARを維持したFRR改善は確認できず、今回の結果は総合的な精度向上を示していない。
これら主条件の差の95% CIはいずれも0を含むため、未知話者一般への改善・悪化を断定しない。
[評価表](evaluation-results.md)、[HTML](evaluation-results.html)、[結果の解釈](evaluation-interpretation.md)を参照する。

評価設定は[evaluation-protocol.json](config/evaluation-protocol.json)の1つに固定する。
モデル2条件・登録各母音10区間・5母音必須・発話内の全適格母音を使う。
現行モデルのhashで凍結された登録・照合結果を再利用し、追加モデルのprofileは同じ登録区間で作り直す。
各モデルのvalidation通常文で目標FAR 1%・0.1%・EER動作点の閾値を決め、
validation別テキストとtest通常文・別テキストへ固定適用する。testで閾値を調整しない。

母音不足は`no_score`とし、本人拒否と全入力分母に残す。全入力・条件付きのFAR/FRR、EER、coverage、
誤判定の実件数を併記する。共有10,000回の話者bootstrapで95% CIと追加−現行の対応付き差を求める。
固定モデル・固定閾値でのCIであり、学習や閾値校正の不確かさは含まない。
既にPhase 3/6/7/8で観測済みのJVS testを使う探索的な評価で、未観測の独立holdoutではない。
匿名client_idとJVS話者の実在人物の重複は確認できない。
登録20/30区間・母音不足対応との組み合わせは今回の2条件に追加しない。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
poc/phoneme-verification-evaluation/.venv/bin/python \
  poc/phoneme-training-data-expansion/run_evaluation.py \
  --output-dir artifacts/phoneme-training-data-expansion/evaluation-20261008-v1
```

未使用の成果物ディレクトリを指定する。入力・コード・モデルをhashで凍結し、validation推論→閾値凍結→
test推論→指標・CI・表の検査→公開の順に実行する。追加モデルは各embeddingを3回反復し、
cacheなしの登録・照合との一致も検査する。現行の全trial・閾値・test指標・CIは既存結果との一致を要求する。
生score、embedding、profile、bootstrap配列、凍結記録はGit管理外の評価ディレクトリへ保存する。

## 検証

```sh
poc/phoneme-speaker-encoder/.venv/bin/python -m unittest discover -s poc/phoneme-training-data-expansion/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check --ignore E402 poc/phoneme-training-data-expansion
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-training-data-expansion
```
