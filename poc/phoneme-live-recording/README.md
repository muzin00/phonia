# 実録音の操作パイロット

既存録音の追加収録シミュレーションで得た改善が、新しい指定文を読む場面でも成立するか確かめるためのローカル試作。今回の範囲はマイク録音、指定文の案内、音素抽出、追加録音の停止、登録プロフィールの保存まで。照合スコア・認証合否・FAR/FRRはまだ表示しない。

元の全36音素encoderと特徴統計を固定し、元のJuliusアライメントと音素抽出を再利用する。新しい指定文5つは全母音をG2P上で2回以上含む。発話による無声化・短縮・録音品質を通過した**実区間**が得られるとは限らないため、取得後の区間数で停止する。forced alignmentは読み間違いを検出するASRでも、生体検知でもない。

## 起動と最初の収録

リポジトリのルートで実行する。既存PoCの仮想環境、Julius、日本語辞書、モデルと特徴統計のローカルartifactが必要。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-recording/server.py
```

[録音画面](http://127.0.0.1:8765)を開き、まず「練習」で1試行を行う。参加者IDには実名を使わず、同じ人は同じID、異なるマイクには異なる端末・マイクIDを使う。同意欄にチェックして準備を完了し、録音開始ボタンを押した後で表示文を読み、読み終えたら停止する。

ブラウザーのマイク権限は録音開始時にだけ要求する。通常のブラウザーからこのPCのlocalhostへアクセスする構成で、スマートフォンからLAN経由で開く機能はない。マイクはOS・ブラウザー設定で選択する。収録は[AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletNode)でPCMを取得し、[getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)の音声処理設定を保存する。エコーキャンセル・ノイズ抑制・AGCは無効を要求するが、実際の設定は機器・ブラウザーに依存する。

## 固定した操作ルール

| 種類 | 停止条件 | 上限 |
| --- | --- | --- |
| 練習・照合用 | 採用済み区間が5母音それぞれ1個以上となった発話境界 | 3回、合計20秒 |
| 登録用 | 20回終了時に各母音10区間以上、プロフィール保存成功 | 20回 |

1回の録音は最大8秒。照合用の残り時間が8秒未満ならその残り時間で停止する。無効区間やアライメント失敗でも、受け取った有効形式のWAVの時間と回数を消費する。WAV形式不正・上限超過など受理前の入力エラーは録音試行へ含めない。同じPCMの再利用は失敗として保存する。成功したテイクだけを選んだり、結果が良くなるまで無制限に録り直したりしない。

16bitモノラルWAVを24kHzへ変換し、元の30〜250ms・RMS −50dBFS基準を使用する。RMS正規化は加えない。登録は音素ごとに音源を分散して最大30区間を選び、平均embeddingを正規化する。「録音完了」は収集条件の達成であり、本人認証の成功を意味しない。

## 保存と追跡

既定の保存先は `artifacts/phoneme-live-recording/pilot-20261011-v2/`。音声artifactはGit対象外。原WAV、24kHz WAV、指定文、アライメント、QC、特徴・embedding、ハッシュ、抽出数、処理時間、失敗理由、同意、申告日・端末・セッションを保存する。端末IDや日付は自己申告で、真正性は検証しない。ブラウザーの機器識別子そのものは保存しない。

初回起動時にプロトコル、実装、モデル、辞書などを `protocol-freeze.json` へ固定する。実装・条件が変わったら `--run artifacts/phoneme-live-recording/<新しい名前>` で別runを作る。以前の録音は上書きしない。同じrunへ複数のサーバープロセスを同時起動しない。処理途中で停止した試行は次回起動時に中断として残す。画面再読込時は同じタブの試行を復元するが、送信前にページを閉じた音声は保存されない。

ファイル入力は `origin=file`、マイクは `origin=microphone`、コーパス再生は別runの `purpose=replay_smoke` で保存する。既存録音やファイル入力を新しいブラウザーマイク収録の件数へ含めない。これは誤集計を防ぐ区分であり、クライアントによるorigin偽装を防ぐ仕組みではない。

```sh
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-recording/pilot_report.py artifacts/phoneme-live-recording/pilot-20261011-v2
```

この集計は成功・失敗・中断、収録時間、処理時間、登録と照合の別日・別端末情報を確認する。認証閾値とFAR/FRRは未校正なのでnullのまま残す。

## 次に人が行う検証

1. 1人の練習で、権限要求、表示文、停止、追加案内、保存まで通るか確認する。マイク拒否・無音・途中停止も記録する。読み切れない文は結果を残し、変更する場合は次runへ分ける。
2. 同じ人の登録20回と、同日同マイクの照合用5試行を収録する。別日に、別の実マイクを選んでさらに5試行を収録する。まずは操作・母音収集の診断であり、1人の結果から他人受入率は推定しない。
3. 操作が安定したら、話者を分離した新しい校正群と評価群、同一マイク／別マイク・同日／別日の条件、single／adaptive1／fixed3の収集割付、必要人数と許容リスクを事前固定する。その後で照合・閾値校正と評価を実施する。現在の画面はadaptive1専用で、正式な3方式比較の収集画面ではない。

JVSのような良好な収録とCommon Voiceのような多様な収録の差を、コーパス名だけで説明しない。次の人による収集では、同じ人のマイク・日付・環境を分けて残し、母音抽出失敗、録音秒数、サーバー処理時間、画面操作を含む経過時間を別々に確認する。照合精度を主張する前に別日・別端末の新しい評価が必要。

## 実装検証

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-live-recording/tests -v
node poc/phoneme-live-recording/tests/audio.test.mjs
poc/phoneme-alignment-evaluation/.venv/bin/ruff check poc/phoneme-live-recording
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-live-recording/smoke.py --run artifacts/phoneme-live-recording/replay-smoke-<未使用名>
```

`smoke.py` は既存JVSの5.02475秒を再処理し、49区間の境界・特徴・embeddingを元cacheと照合する。同一JVS話者の異なる20音声で登録保存も検証する。新しい人の録音は生成しない。再生確認の詳細は[検証記録](validation.md)を参照。
