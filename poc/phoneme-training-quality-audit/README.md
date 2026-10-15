# 学習データ品質の全件調査

2026-10-16、Common Voice除外比較の完了後に、現行の学習入力と抽出工程を調査した。
**基本的な前処理・除外は存在するが、文字起こし、音素境界、波形品質の検査には不足がある。**
採用済みデータにも追加確認が必要な実例が残っている。今回の検出件数は不良率ではなく、
これらがEERを悪化させたか、除外すれば改善するかは未検証である。

## 対象と実行範囲

- 対象: `artifacts/phoneme-corpus-expansion/src490-20261013-v1/train-segments.jsonl`
  と対応する128次元特徴量、元WAV、Julius alignment。
- 全1,930,593学習区間・37,874録音・490話者ラベルを検査した。
  話者ラベル数は実在人物の同一性を保証しない。
- JTubeの採用元10,027録音について、保存済みWhisper結果も全件検査した。
- 調査時のベースcommit: `da7ec95cbda3df9ce4bfa2c4dd2709b724a8566b`。
  調査ブランチ: `investigate/training-data-quality`。
- 学習、ASR再実行、音声の加工・差し替え、凍結済み実験の変更は実施していない。
  評価セットの品質検査や人間による聴取判定も、この調査には含まれない。

| 学習コーパス | 採用区間 | 元録音 | 話者ラベル |
| --- | ---: | ---: | ---: |
| Common Voice | 666,180 | 15,263 | 140 |
| JTube | 508,771 | 10,027 | 210 |
| JVS | 585,772 | 9,094 | 70 |
| SRC4VC | 169,870 | 3,490 | 70 |
| 合計 | 1,930,593 | 37,874 | 490 |

## 現在存在する処理

1. 24kHz・mono・PCM16への変換／形式検査、音声hashとメタデータの対応検査。
2. 提供テキストまたはASR結果をG2Pで音素列へ変換し、Juliusで強制アライメントする。
   失敗した発話、対象外音素、pauseを除外する。
3. 区間が音声範囲内かを検査し、30ms未満またはRMSが−50dBFS未満の区間を除外する。
   250msより長い区間は中央250msへ切り詰める。
4. 特徴抽出時にDC成分を除き、log-Mel特徴量を学習統計で標準化する。
   現在の呼び出しは `rms_enabled=False` であり、波形のRMS音量正規化は無効。

実装根拠:
[区間抽出と品質フラグ](../phoneme-greedy-selection/prepare.py)の`extract`、
[特徴抽出の設定](../phoneme-greedy-selection/common.py)の`pipeline`、
[DC除去と特徴標準化](../phoneme-speaker-encoder/phase3_data/input.py)の`InputPipeline`、
[追加データの採用](../phoneme-corpus-expansion/prepare.py)。

ただし、強制アライメントの成功は、テキストが音声と合っていることの独立した検証ではない。
現在の採否には、ASRの低信頼度・反復指標、alignment score、音割れ率、区間内のゼロ値率を使用していない。
alignment scoreは元ファイルに残るが、採用manifestの品質判定には引き継がれていない。

## 全件検査で確認できたこと

ファイル読み込み・形式、hash、区間参照、元音素列とのラベル対応、採用時の長さ・音量規則、
特徴量の有限性・ファイルサイズ、重複区間を検査し、不整合は検出されなかった。
採用元WAVのPCM全体が完全一致する重複も0件だった。
これは近似重複や別録音中の同一人物、音声に対するラベルの正しさを保証する検査ではない。

以下は**追加確認用のスクリーニング件数**。条件は重複し、合算できない。
すべて実際に採用された区間の波形で測定した。

| 検査条件 | Common Voice | JTube | JVS | SRC4VC | 合計 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最大振幅近傍のサンプルが1%以上 | 2 | 0 | 0 | 41 | 43 |
| DC除去後のRMSが−50dBFS未満 | 90 | 321 | 22 | 0 | 433 |
| サンプルの過半数が完全なゼロ | 127 | 26 | 1 | 449 | 603 |
| 切り詰め前の1音素区間が1秒超 | 10 | 237 | 7 | 8 | 262 |

- 最大振幅近傍は `abs(PCM / 32768) >= 0.999`。実際のPCM16上限・下限に
  1%以上張り付く区間は29件で、すべてSRC4VCだった。録音時の歪み全般を検出するものではない。
- ゼロ値は聴感上の無音判定とは異なる。子音の閉鎖区間など、弱い信号が妥当な場合もある。
  SRC4VCの過半数ゼロ候補には `a` 83件、`o` 57件、`i` 39件、`e` 39件も含まれる。
- 1秒超は境界を再確認するための指標。長母音などの正しい区間もあり、全件を誤ラベルとはしない。
- 単に振幅が−50dBFS未満のサンプルが過半数という条件では201,764区間が該当した。
  JVSにも70,936区間あり、この条件だけで除外すると正しい低振幅音素まで失う可能性がある。
- コーパス別の母数は異なる。この表から「Common Voice全体の品質が悪い」とは結論できない。
  主要な検出条件の該当数は全193万区間に比べ小さく、検出されない誤ラベルや雑音は別途残り得る。

## 確認すべき具体例

元WAVへのパス、フレーム位置、区間ID、測定値は
[`focused-examples.json`](../../artifacts/phoneme-training-quality-audit/src490-20261016-v1/focused-examples.json)
に保存した。表のcache indexは学習manifestの0始まりの行番号に対応する。
これらの極端な例は元WAVを再読込し、累積和を使わない直接計算でも数値を確認した。

| cache index | コーパス・音素 | 実測と処理上の問題 |
| --- | --- | --- |
| 1,308,705 | CV・`d` | 採用時RMS −47.89dBFSがDC除去後−69.62dBFSになる。学習時に取り除くDC成分を、採用時の音量判定では含めている。 |
| 1,859,972 | SRC4VC・`t` | 30ms中の15.42%が最大振幅近傍。実際のPCM上下限への張り付きもある。 |
| 1,918,940 | SRC4VC・`i` | 採用した250msの90.62%が完全なゼロ。RMSは−43.46dBFSで現行基準を通過する。 |
| 1,124,924 | JTube・`g` | 元alignmentは1音素へ8.62秒を割り当てている。中央250msへの切り詰めで最終的な長さ条件は通るが、境界の妥当性は確認されない。 |

元録音の問題と切り出しの問題を分けるには、採用区間と前後の原音声の照合が必要。
この調査では聴取結果を作成していない。

## JTubeの文字起こしに残る検査の不足

保存済みASR診断値を確認した結果:

| 条件（録音内に1つでも該当するASR segment） | 録音数 / 10,027 | その録音からの採用音素区間数 |
| --- | ---: | ---: |
| `avg_logprob < -1` | 144 | 2,484 |
| `compression_ratio > 2.4` | 6 | 324 |

右列は該当録音由来の**全採用区間**であり、誤ラベルと確認された区間数ではない。
これらの閾値は使用中のMLX Whisperがデコード再試行の判定に使う値を参照した。
品質の正解ラベルや、校正済み誤認識確率としては扱えない。
保存された診断値はデコード窓に由来し、最終的な短いテキスト断片だけの指標とは限らない。

具体例として、`jtube_avr`の7.344秒の録音で、文字起こしが「4」に「〜」222個を付けた
223文字となっている。compression ratioは39.24だが、採用から除外されていない。
G2Pでは`y o N`となり、3音素区間が採用された。異常な文字列が222音素に展開されたわけではない。
音声に対する3ラベルの正誤自体は聴取での確認が必要。
元ASRファイルは
`artifacts/phoneme-corpus-scaling/expanded-inputs-20261011-v1/asr/7a170affe4004f11165bc4db7d93230dcf818846eaf54bfaa24aa19404e30078.json`。

コード上の経路:

1. [corpus_asr.py](../phoneme-corpus-scaling/corpus_asr.py)と
   [expansion_asr.py](../phoneme-data-scaling/expansion_asr.py)は、文字列が空でなければ
   `transcribed`として保存し、ASR診断値による学習採用の保留・除外を行わない。
2. 両configの `temperature` はスカラーの `0.0`。
   使用環境の `mlx_whisper/transcribe.py` の `decode_with_fallback` はスカラーを1要素リストへ変換する。
   低logprob・高compression ratioで再試行フラグが立っても、次のtemperatureがなく、その結果を返す。
3. MLX Whisper内部にはno-speechによるskip処理が存在する。
   「ASR内に品質関連の処理が一切ない」という意味ではない。
   今回の採用元で保存診断値 `no_speech_prob > 0.6` の該当は0件。

したがって、ASRの失敗候補をデータ採用へ反映する独立した検査が必要。
再試行を有効にするだけでも正しい文字起こしを保証できず、再試行後の採否記録は別途必要になる。

## 次に実装・検証する方針

この調査の結果から、コーパス一括除外より、以下の順で品質規則を検証するのが妥当と考える。

1. **学習時と同じ波形処理の後でRMSを検査する。**
   DC除去前後のRMSを記録し、現在の採否と学習入力の食い違いを解消する。
2. **録音単位と音素区間単位の品質情報を保存する。**
   ASR診断値・テキストの反復、波形の上下限への張り付き、ゼロ率、切り詰め前の長さ、
   alignment scoreをmanifestへ対応づけ、採用・保留・除外と理由を追跡できるようにする。
3. **文字起こしと境界の誤りを優先して確認する。**
   今回の極端な候補に加え、フラグが立たない区間もコーパス・音素別に抽出し、見逃しを調べる。
   録音自体が使えないのか、テキストや切り出しを直せば使えるのかを区別する。
   長さやalignment scoreの一律閾値は、正しい長母音・短い子音を除きすぎないか確認して決める。
4. **規則を固定した新版データで、品質除外の効果を比較する。**
   元データ、品質規則による除外、同程度の量をコーパス・話者・音素構成を可能な範囲で揃えて
   ランダム除外した対照を用意する。更新数と初期値を揃え、データ量減少と品質効果を分けて調べる。
   残るsingletonや話者数の変化も記録する。既に観測したtestは探索的評価とし、可能なら未使用評価で確認する。

今回の閾値をそのまま自動削除規則として採用していない。
除外規則の選択はtrainの品質証拠で行い、test EERに合わせて個々の区間を選ばない。
前回の[CV除外比較](../phoneme-common-voice-ablation/RESULTS.md)でも、CV全除外による
全体EERの改善は確認できておらず、今回の調査もCV全体の不良を裏付けるものではない。

## 成果物と再現方法

このディレクトリの [`summary.json`](summary.json) は、全件検査とASR検査の主要集計の保存版。
詳細は `artifacts/phoneme-training-quality-audit/src490-20261016-v1/` にある。

- `results.json`: コーパス・音素別集計、構造検査、入力hash。
- `segment-measures.npz`: 全区間の測定値。配列行は元manifestのcache indexと一致する。
- `source-measures.jsonl`: 元録音の検査とhash。行番号はNPZの`source`値に対応する。
- `review-candidates.json`: 条件ごと・コーパスごとの先頭候補と低alignment score候補。
  無作為標本ではなく、不良率推定には使用できない。
- `focused-examples.json`: 表で挙げた極端例と元波形からの直接再計算値。
- `asr-results.json`, `asr-diagnostics.json`: 保存ASRの集計・録音別診断値とテキスト。
- `complete.json`: 波形・特徴検査の入力／出力hash。
- `analysis-complete.json`: 追加ASR検査・例示・保存版集計と解析コードのhash。

最初の実行はリポジトリrootから以下の順で行った。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality-audit/audit.py
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality-audit/asr_audit.py
```

`audit.py` は完了済み出力があれば停止する。再検査する場合は旧結果を残し、
`OUTPUT`を新しいrun名に変えた別版で実行する。ソースデータと出力先は本調査用に固定している。
検査にはNumPyを使用し、PyTorchの学習処理は起動しない。

測定処理の単体テストは、直接計算との一致、DCだけで採用時音量を通過するケース、
PCM16負側の最大値のオーバーフロー防止を検証する。

```sh
poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-training-quality-audit/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-training-quality-audit
```
