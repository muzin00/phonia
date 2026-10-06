# ECAPA-TDNNのJVS validation動作確認

## 結果

2026-10-06に、固定したECAPA-TDNNでvalidationの5話者・11発話すべてから
192次元の有限・非零embeddingを取得した。各発話の3回の推論はbit単位で一致し、
別プロセスでの再実行でも全11件のembeddingと選択入力が一致した。
モデルのパラメータ・buffer、取得した重み・設定ファイルも実行前後で不変だった。

この確認はembedding抽出の成立性と再現性を対象にしたものである。
登録profile作成、認証閾値の較正、FAR・FRR・EERの比較測定はまだ行っていない。

## 固定した条件

| 項目 | 条件 |
| --- | --- |
| モデル | `speechbrain/spkrec-ecapa-voxceleb` |
| revision | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| 入力 | Phase 6と同じJVS元WAV、validationのみ |
| 選択 | jvs007・jvs016・jvs021から登録・通常文・異文を各1発話、通常文／異文の最短発話を追加 |
| 追加した最短発話 | 通常文: jvs070、3.538875秒。異文: jvs045、2.272458秒 |
| サンプルの発話長 | 約2.272〜8.748秒 |
| 波形変換 | mono PCM16 / 32768.0、float32、torchaudio sincによる24→16 kHz変換 |
| resampler | `sinc_interp_hann`、filter width 6、rolloff 0.99 |
| 前処理 | モデル付属の特徴抽出・発話内平均正規化。VAD・crop・RMS正規化は追加しない。 |
| embedding | `encode_batch(..., normalize=False)`。保存されたembedding統計による正規化は行わない。 |
| 推論 | CPU / float32 / batch=1、torch thread=1、interop thread=1、決定論的アルゴリズム、勾配無効 |
| 環境 | macOS 26.6.2 / arm64、Python 3.12.13 |
| 主な依存関係 | numpy 2.5.3、torch/torchaudio 2.11.0、SpeechBrain 1.1.1、huggingface-hub 2.1.1、scipy 1.18.1 |

詳細は[設定](config/embedding-smoke.json)と`uv.lock`へ保存した。
torch/torchaudioを同じリリースへ揃えるため、torch 2.14.1のPhase 6とは異なる専用環境を使用した。
Phase 6のコード・環境・成果物を変更していない。

## CPU計測

以下は主runの値。モデル読込と最初のwarmupは発話ごとの推論時間から分離した。
時間は1回の小規模動作確認での値であり、別ハードウェアでの速度を保証しない。

| 項目 | 値 |
| --- | --- |
| モデル読込 | 約84 ms |
| 最初のwarmup | 約45 ms |
| 平均推論時間 | 約43.83 ms/発話 |
| 平均real-time factor | 0.00793 |
| プロセス最大RSS | 758,546,432 bytes（約723.4 MiB） |
| 別プロセスの平均推論時間 | 約44.45 ms/発話 |

推論時間は各発話3回の平均をさらに11発話で単純平均した値で、WAV読込・checksum検査・
resamplingを含まない。real-time factorも各発話の値の単純平均である。
最大RSSはPython・依存ライブラリ・モデル・一時tensorを含むプロセス全体のピークであり、
モデル単体のメモリ使用量ではない。

## 保存した記録と検証

主runは`artifacts/phoneme-baseline-comparison/ecapa-smoke-20261006-v2/`、
別プロセスでの再確認は`ecapa-smoke-20261006-v2-recheck/`へ保存した。
モデルsnapshotは`artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0/`。
各runに選択した入力、embedding、反復差、計測値、モデル・コード・lockのchecksumを保存した。

| 記録 | SHA-256 |
| --- | --- |
| 主runの`smoke-report.json` | `aa524013b772556c0ec3aff10f45ec1c7d087c93ffed7c6db1b712a189167464` |
| 再確認runの`smoke-report.json` | `ae556ec74e4a3f461fd3683fe859c870d214d99c415e7ba5bf429a282c040a8f` |
| 両runの`embeddings.npy` | `aafd7dd17cb7925de83b1b66ad183d2503b4dfd9aa92debb93e9f8af58d92038` |
| モデルsnapshotの`model-manifest.json` | `c5c539860b40e98531296d61d42ffd8147f51bad0b91aed05e76c7608b757cec` |

別プロセス間の検証は主runの`cross-process-check.json`へ保存した。
全8件の入力・checksum保護テスト、ruff check、ruff format --check、git diff --checkが通過した。
旧v1はchecksumの設定固定前の初回動作確認として保持し、ここでは現在の実装と一致するv2を報告する。

## 次の工程

発話全体方式のembedding抽出を実装する環境と重みが揃った。
次は各母音1・5・10区間の登録と発話全体方式の登録音声量の対応、queryで利用する音声量、
文脈方式、入力不足の扱いを比較設計へ定義する。

今回の最短入力は約2.27秒の発話である。音声量を揃えるためにさらに短く切る条件や、
30 ms単位の母音をこのモデルへ入力する条件の成立性は別に確認する必要がある。
認証性能、日本語への一般化、音素分割の効果はこの動作確認だけから判断しない。
