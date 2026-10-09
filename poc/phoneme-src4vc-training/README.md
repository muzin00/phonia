# Phase 8: 別コーパスSRC4VCで学習する

[Issue #47](https://github.com/muzin00/phonia/issues/47)の追加検証。
`feature/phase8-src4vc-training`でデータ準備・30,000 updateの学習・JVS validationを完了した。
同じ30,000 updateのCommon Voice版と比べ、validation macro EERは9.228→8.887%だった。
1 seedでの観測値で、属性の偏りが改善不足の原因だったとは特定できない。
続く[発話単位評価](../phoneme-src4vc-evaluation/README.md)も完了した。
CV版との全入力FAR比較は通常文2.657→1.714%、別テキスト2.000→1.413%。
全入力FRRは通常文2.933→2.400%、別テキスト13.556→13.778%で、差の95% CIはいずれも0を含んだ。
Common Voiceを含むモデルの学習回数を揃えてもFAR改善を確認できなかったため、別の日本語コーパスを試す。
新規条件はJVS 70話者＋SRC4VC 70話者の1つ、seedは20260926の1つに固定する。
設定は[protocol.json](config/protocol.json)の1つ。条件の追加や学習率の探索は行わない。

## 選定したデータと比較条件

[SRC4VC公式配布](https://y-saito.sakura.ne.jp/sython/Corpus/SRC4VC/index.html)のver1を使う。
日本語100話者のスマートフォン実録音で、話者ID・テキストが揃っている。研究目的で無償、音声の再配布は禁止。
原録音の読み上げ・感情発話・対話調を使用し、歌唱と復元音声は除く。
各話者の読み上げは10発話だけなので、話者表現の学習機会を確保するため、歌唱以外の最大50発話を使う。

100話者を固定seedのhashで並べ、70話者をtrain、15話者を今後のvalidation、15話者を今後のtest用に割り当てる。
発話数の多い話者を優先せず、利用可能な100話者全員を選択候補にする。
録音の品質・年齢・性別による選別や評価結果による選び直しは行わない。取得できる属性は偏りの確認に保存する。
今回はSRC4VCの予約30話者の音声を、区間抽出・特徴統計・学習・validationに使用しない。
JVSの70/15/15分割は保持する。両コーパスの実在人物の重複は未確認であり、ラベル分離を人物同一性の保証とは扱わない。

encoderは65,920 parameters・128次元の`statistics_mlp`、140ラベルの分類head、損失・batch・前処理は従来と同じ。
同じseedの初期化から30,000 updateまで学習する。Common Voiceの重みから継続する方式ではない。
比較対象は既存のJVS＋Common Voiceモデル30,000 updateで、両方の採用時点を固定する。
元のwarmup 1,000 update・最大30,000 updateのcosine scheduleを維持し、early stoppingは無効にする。
JVS各話者のbatch選択回数が現行JVSモデルの採用時点と最大1回差以内になることを完了条件とする。
JVS validationは途中経過を記録するが、checkpointの採用時点を変えない。

JuliusとG2Pによる母音境界抽出、30 ms未満・`silent`・`near_silent`の一律除外を引き継ぐ。
失敗と除外を全件記録する。特徴統計はJVS trainとSRC4VC trainだけで計算する。
追加データの量・話し方・収録条件・特徴統計が同時に変わるので、人口属性の偏りだけを切り分ける実験ではない。
この学習protocolは学習とJVS validationまでを対象にし、発話単位FAR/FRRは後続の独立した評価protocolで計算した。
学習段階の報告では`test_used=false`を維持する。後続の評価結果は[HTML比較表](../phoneme-src4vc-evaluation/evaluation-results.html)へ分けて保存した。

成果物はGit管理外の`artifacts/phoneme-src4vc-training/src4vc-20261009-v1/`へ保存する。
元の学習・評価・データの正本は保持する。

学習完了後の入力・出力checksumの再検証を通過した。
JVS各話者のbatch選択回数は2,142〜2,143回で、従来JVSモデルと各話者最大1回差以内。
encoderを保存して再読み込みし、validation 8区間のembeddingがbit単位で一致することを確認した。
[結果のHTML比較表](training-results.html)と[詳しい集計](training-results.md)を参照する。

## 準備の実測

| 追加データ | 追加話者ラベル | 選定発話 | 抽出成功 | 学習可能な母音区間 | 原音声時間 | 母音時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common Voice（既存比較対象） | 70 | 8,283 | 8,280 | 198,446 | 11.509 h | 3.721 h |
| SRC4VC | 70 | 3,500 | 3,499 | 91,358 | 7.252 h | 1.964 h |

SRC4VCの除外は13,787区間、失敗はJuliusの候補なし1発話だった。失敗した発話の代替選定は行っていない。
全70話者で5母音それぞれ2区間以上の学習候補がある。
JVSを含む総学習区間は415,271件となり、Common Voice版522,359件より少ない。
話者のbatch選択回数を揃えても、各区間の反復回数や収録時間までは一致しない。

SRC4VC trainの申告属性は女性43・男性27、21〜68歳（平均39.2歳）。属性を使って話者を選び直していない。
既存Common Voiceの選定70ラベルは男性41・女性23・未回答6で、20代31ラベルだった。
これは元データの自己申告属性の集計で、年齢・性別による精度差や改善しなかった原因を証明するものではない。
Common Voiceの属性集計は`artifacts/phoneme-training-data-expansion/selection-demographics-20261009.json`に保存した。

発話IDの大文字と既存alignment処理のID形式が合わなかった初回準備の記録は、成果物内の
`preparation-attempt-id-format/`へ保持した。小文字IDに修正後、同一の話者・発話を処理し直した。

## 実行

依存環境は既存のalignment環境（`pyopenjtalk==0.4.1`、`PyYAML==6.0.3`）とspeaker encoder環境を使用する。
今回の学習runtimeはPython 3.12.13・PyTorch 2.14.1・NumPy 2.5.3・CPU 1 thread。
公式ZIPを`artifacts/phoneme-src4vc-training/src4vc-20261009-v1/source/SRC4VC_ver1.zip`に取得する。
サイズ検証・SHA-256固定後に次を順に実行する。元音声は再配布しない。

公式配布にchecksumがないため、取得したZIPのSHA-256を選択前に保存し、再実行時には同一性を検証する。
配布元が同じURLの内容を更新した場合は、保存済みmanifestとZIPに基づいて再現する必要がある。

```sh
poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-src4vc-training/prepare_src4vc.py
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-src4vc-training/train_src4vc.py statistics
poc/phoneme-speaker-encoder/.venv/bin/python poc/phoneme-src4vc-training/train_src4vc.py train
python3 poc/phoneme-src4vc-training/build_report.py
```

準備は各発話の結果を保存する。学習は1,000 updateごとのcheckpointからoptimizer・scheduler・RNGを復元できる。
同じ設定で完了済みの学習を再実行すると、凍結入力と成果物のchecksumを検証して終了する。
数値は[学習比較表](training-results.html)、[Markdown](training-results.md)、[JSON](training-results.json)へ出力する。

```sh
MPLCONFIGDIR="$PWD/artifacts/phoneme-training-data-expansion/matplotlib-cache" poc/phoneme-alignment-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-src4vc-training/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-src4vc-training
poc/phoneme-verification-evaluation/.venv/bin/ruff format --check poc/phoneme-src4vc-training
```
