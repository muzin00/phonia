# 全音素からランダム追加してencoderを選ぶ

5母音を固定し、候補の箱からランダムに1音素取り出して学習する。
validation通常文の統合EERが現在の最良値より下がれば、その音素とencoderを採用する。
同値・悪化なら採用せず、最良encoderを保持する。
採用セットが変わった後は、不採用候補を元のランダム順で再試験する。
1巡で採用がなければ探索を終了する。全候補を採用した場合も終了する。

探索する組み合わせは、各時点の採用セットに候補を1つ追加したもの。
全候補は箱に含めるが、全組み合わせの総当たりは行わない。
同じ音素セット・seedの学習と測定が既に完了している場合は、その結果を再利用する。
採用セットが変わって新しい組み合わせになる場合に、初期値から再学習する。

## 候補の範囲

**9子音などのホワイトリストは設けない。**
JVSとCommon VoiceのG2Pラベルに存在する全音素を候補とする。
撥音N・促音cl・拗音・半母音・v/ty/dyなどの少数音素も残す。
無声化母音A/I/U/E/Oは固定5母音へ含め、休止pau・silB/silEは音素とは数えない。
候補は`inventory.json`に固定し、区間不足の候補も一覧から消さない。
元ラベルv/tyはアライメント内部でb/chへ変換されているが、候補ラベルは元のv/tyのまま保存する。

JVS train 70話者と、以前選んだCommon Voice日本語train 70話者を使用する。
SRC4VCは使用しない。原境界内の30〜250msを中央cropし、RMS −50dBFS未満を除外する。
切片の無音pad・増幅・文脈追加は行わない。除外区間と理由を別manifestに保持する。
各音素は、2つ以上の異なる適合区間を持つ話者から最大10話者を選ぶ。
全140話者でその音素が揃うことは要求しないため、少数音素も学習できる。
適合する区間ペアが1話者もない場合は「データ不足」として残し、EERを捏造したり性能上の不採用と扱ったりしない。

## 学習と採否

各候補セットを同じseed・同じencoder/head初期値から30,000更新で学習する。
encoderは既存のstatistics pooling＋MLP（65,920 parameters、128次元）。
AdamW、AAM-Softmax＋音素内SupCon、warmup/cosine scheduleは既存実装と同じ設定を使う。
各更新は5つの音素group。通常は10話者×2区間×5group＝100区間。
少数音素は利用可能な話者数だけを使い、欠けたgroup内slotは損失とSupConの分母から完全に除く。
固定した最終更新の重みを採用し、途中checkpointでvalidation成績を最大化しない。

時間を短縮するため、学習されないlog-Mel処理とstatistics poolingを入力ごとに1度計算する。
元のencoderのprojectionをそのまま学習し、通常の波形入力で使えるencoderとしてexportする。
cache推論と通常の波形入力、vectorized lossと元の損失・勾配の一致を検査する。
固定した中央cropは全候補で共通。特徴正規化には既存のtrain限定5母音統計を全条件で共通使用する。

## 評価

同じvalidation/test各15話者、通常文750・別テキスト450発話を使う。
登録元WAVのプールは既存実験と共通で、各音素について最大10区間を発話分散して選ぶ。
同じ音素の登録・照合切片は全候補で不変。音素を追加しても既存の母音を短くしない。
追加音素の音声分だけ実使用時間が増えるため、固定時間当たりの優劣を測る実験ではない。

照合は5母音必須で、追加音素が発話にある場合に使用する。
ある追加音素を使うのは、全15登録話者にその音素のprofileがある場合だけ。
同じ照合発話では全claimed speakerで同じ音素セットを使用する。
登録できない少数音素も学習候補には残り、そのモデルは利用できる登録音素で評価する。
音素ごとのquery区間と登録平均のcosineを平均し、最後に音素間を等重みで平均する。
5母音不足は全候補で同じno_scoreとし、EERはscoreのある同じ発話集合で算出する。
FAR/FRRではno_scoreを含む全入力の分母も併記する。

探索にはvalidation通常文の統合EERだけを使う。
最後の選択セット、encoder、validation校正閾値を固定してから、基準モデルと最終モデルだけをtestで測定する。
途中候補のtest測定や、testの成績による採否の変更は行わない。
JVS testは過去に使用した集合であり、新しい独立holdoutではない。
この探索は良いencoderと組合せを探す目的で、各音素固有の普遍的な寄与ランキングを確定するものではない。

## 実行

設定は[`config/protocol.json`](config/protocol.json)の1ファイル。
出力は`artifacts/phoneme-greedy-selection/all-phones-20261011-v2/`。
v1は入力準備だけを行った。学習開始前に監査・公開用ツールを完成させ、入力をbyte単位で保持したままv2へ固定する。
入力準備は未使用ディレクトリにだけ実行する。探索はoptimizer/RNG checkpointから再開できる。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-greedy-selection/tests -v
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/prepare.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/freeze_sources.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/study.py verify-inputs
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/study.py search
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/study.py test
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/audit.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/phonia-mpl poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-greedy-selection/report.py
```

学習前の候補試聴サンプルも`listening/`へ用意する。
各コーパス・音素から1区間、hash順で選び、保存なしの既存レビューUIで再生できる。
正式レビューや回答保存は依頼されていないため、回答JSONLや品質OKファイルは作成しない。

完了後の結果は`evaluation-results.html`・`evaluation-results.md`・`interpretation.md`で確認する。
