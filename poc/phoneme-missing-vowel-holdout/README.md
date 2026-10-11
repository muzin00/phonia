# 母音不足対応の新規話者holdout評価

前回のJVS探索から、**全36音素encoder・登録上限30区間・等重み・4母音以上**を主候補に固定する。5母音必須を対照、3母音以上かつ合計5音素を記述用の補助条件とする。encoderや統合方法の学習、testを見ての方式変更は行わない。

手元に固定済みのCommon Voice 17日本語アーカイブを使用し、既存学習70 client_idを除いた50発話以上の話者から、固定hash順で60話者を選ぶ。先頭30話者を校正、残り30話者をtestへ割り当てる。各話者は固定hash順で50発話、最初20発話を登録、残り30発話を照合へ使用する。元コーパスのsplit名はtrainだが、今回の話者はプロジェクトの学習・従来評価には使っていない。client_idと実在人物の対応やJVSとの人物重複は保証できない。

選択した話者・発話・処理コード・モデル・元データhashを新規音声処理前に固定する。校正側を先にアライメント・特徴抽出・採点し、全条件の閾値を凍結してからtest側を処理する。前回と同じJulius・G2P・30〜250msの境界内中心crop・RMS−50dBFS以上・固定されたtrain特徴統計を使用する。登録は録音元を分散させて音素別最大30区間、照合は全利用可能区間。各集合で全登録話者に存在する学習済み音素だけを共通利用し、候補話者に依存しない支持集合にする。

各方式について、新校正30話者で決めた目標FAR 1%・0.1%・EER動作点と、前回JVS由来の閾値を移した診断を別々に報告する。入力不足・アライメント失敗も全入力の拒否として残し、都合のよい音声へ差し替えない。選択集合内の圧縮音声hash重複は全出現を除外する。PCM変換後も各split内の重複を全件除外し、既存学習・評価音声または先行する校正音声と重なるtest音声も除外する。照合slotは拒否として保持する。

## この実験で結論を出す範囲

「次のプロトタイプの研究候補へ4母音方式を採用する」ための仮の判定基準を、新testを見る前に次の3点へ固定する。

- 4母音方式のtest全入力FARの話者bootstrap 95%区間上端が1%以下。
- 5母音方式に対する全入力FAR増加の95%区間上端が0.1 percentage point以下。
- 全入力FRR差の95%区間上端が0未満。

0.1 ppは既存の1%目標の1/10を追加受入の上限に置いた、今回の研究用判断値であり、利用者から承認された本番のリスク許容値ではない。条件を満たさない場合は「現構成の採用は保留」と結論づけ、同じtestで探索を続けない。区間は固定モデル・固定閾値に条件付いた話者再抽出であり、校正不確実性やゼロ誤受入の真の上限を保証するものではない。

録音セッション情報がないため、別日・別端末の実利用性能、なりすまし耐性、本番認証への採用はこの実験の判定対象に含めない。新規学習は行わないため、学習データの採用レビューには当たらない。

## 再実行

リポジトリrootから実行する。既存データと依存環境を使う。`freeze`は未使用runのみ許可。再実行時は各コマンドに同じ新規`--run`を付ける。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-missing-vowel-holdout/study.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-missing-vowel-holdout/study.py calibrate
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-missing-vowel-holdout/study.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-missing-vowel-holdout/study.py report
```
