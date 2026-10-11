# 現在の登録プロフィールと既存他人候補音声の照合

本人の類似度が高いだけで認証成功と判断せず、同じ登録プロフィールに他人候補の音声を照合してスコアの重なりを調べる。新しい他人音声の取得は困難という利用者の回答により、既存Common Voiceの2集合を参考評価に使う。機器原因の調査は行わない。

モデル、本人の20録音プロフィール、5母音必須、音素平均の等重みスコアを固定する。Common Voiceにある他の登録プロフィールは使わず、すべての声を今回の1人の登録に対して採点する。各入力で、今回の登録に存在してその入力でも取得された音素を使う。照合側の平均embeddingを再正規化しない。

## 比較する入力

- CV-A / CV-Bはそれぞれ既存のvalidation 30・test 30 clientラベル。各人の照合用30発話を全数使用し、登録用の既存発話は使わない。合計120ラベル、校正60・参考test60ラベル。clientラベルと実在人物は同一とは限らず、今回の本人との実在人物重複も未確認である。
- `single`は各発話を単独で採点する。
- 主条件`adaptive1`は同一client内の固定hash順で各発話を開始点に、最大3発話を連結する。5母音がそれぞれ1区間以上となった最初の発話境界で終了する。900開始点/集合/分割を残すため、同じ音声が複数episodeに現れ、900の独立試行とは扱わない。
- コーパス側は元の録音全体・元の音素区間を使う。マイク版の1録音8秒・合計20秒へ加工しておらず、制限内に収まる件数を併記する。文章・話し方・収録環境も本人入力とは異なる。
- 本人側は収録済みの全照合episodeを残す。最初の1録音と、実際に追加された全録音を診断する。音声あり中断・録音前中断も未採点として残し、登録や練習音声を本人照合へ流用しない。

## 閾値と評価

本人スコアへ合わせず、CV-A/B validationの採点できた他人候補スコアをまとめ、経験的受入率1% / 0.1%以下になる参考閾値を各armで決める。同点はまとめて扱い、境界値を厳密に上回る閾値を選ぶ。この閾値をファイルへ固定してから、既存test分割と収録済み本人入力へ適用する。test分割も以前の研究で既観測なので、新規独立holdoutとは呼ばない。

結果には採点可能数、全入力・採点可能入力それぞれでの他人候補受入、少なくとも1回受け入れられたclient数、本人の受入・未採点を実件数で示す。本人最低スコアを上回る他人候補の件数は分布の重なりの診断として報告し、その境界を運用閾値にはしない。本人は1人・少数・同日であり、FRRや人物横断の精度を推定しない。

## 再現性と制約

旧実験で固定したキャッシュのハッシュを検証し、全embeddingを元特徴量と固定encoderから再計算する。今回のプロフィールと本人スコアも元結果と照合する。元のデータ・モデル・マイクサーバー・登録条件は変更しない。既存他人候補と本人が分離しても、環境差や文章差による見かけの分離が残る。同じ条件の新しい他人音声がないため、運用上の認証成功や安全性を保証しない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-live-impostor/tests -v
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-impostor/screen.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-impostor/screen.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-impostor/screen.py test
```
