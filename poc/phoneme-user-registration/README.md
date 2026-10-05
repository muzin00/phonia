# Phase 4: 固定encoderによるユーザー登録

## 目的と現在の状態

[Issue #38](https://github.com/muzin00/phonia/issues/38)の登録・保存・読込を実装した。
学習済みモデルへ登録用の母音音声を入力し、ユーザーごとの母音別代表ベクトルを保存する。
Phase 5は保存済みプロファイルを読み込み、別の音声の同じ母音の特徴と比較できる。
照合処理と初期のスコア統合は[Phase 5](../phoneme-verification/README.md)に実装した。
登録時に学習、optimizerの構築、trainデータの読込、特徴統計の再計算は行わない。

Phase 4のコード、設定、テスト、Python依存関係は、この`poc/phoneme-user-registration/`
に置く。Phase 3の`InputPipeline`とencoder実装は隣の`phoneme-speaker-encoder/`から参照する。
採用済みcheckpoint・特徴統計はPhase 3の成果物を使用する。

## 環境構築と検査

Python 3.12を使用する。推論の再現確認に使ったNumPy / PyTorchの版を固定し、
ruffを`dev`依存関係に含める。`uv sync`は開発用依存関係もインストールする。

```sh
uv sync --project poc/phoneme-user-registration

uv run --project poc/phoneme-user-registration \
  python -m unittest discover -s poc/phoneme-user-registration/tests -v

uv run --project poc/phoneme-user-registration \
  ruff check poc/phoneme-user-registration

uv run --project poc/phoneme-user-registration \
  ruff format --check poc/phoneme-user-registration
```

対象はJulius等で抽出済みの5母音区間である。録音音声からのアライメントは
[Phase 1](../phoneme-alignment-evaluation/README.md)の処理を先に実行する。
録音UI、アライメントを含む一括登録、本人照合・スコア統合・閾値決定、オンライン更新は
この実装の対象に含めない。

## 固定encoderと前処理

- 採用構成: `log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel`
- 単一配布用seed: `20260926`、70話者で学習した選択checkpoint
- encoder: `statistics_mlp`、出力128次元
- 正本: `config/registration-policy.json`。最終test計画のcheckpoint・run・特徴統計の
  SHA-256を転記し、読込時に検証する。
- 元WAV: 24 kHz、モノラル、signed 16-bit PCM。自動resamplingや形式変換は行わない。
- Phase 3の`InputPipeline`をそのまま利用する。30 ms以上、250 ms超は中央crop、
  DC除去、RMS正規化なし、64-bin log-Mel、保存済みtrain特徴統計で標準化する。
- CPU / float32 / batch=1、evalモード、勾配無効、全parameterの`requires_grad=False`。
  数値の再現確認は同じ実行環境で行う。異なるCPU/PyTorch版でのビット一致は保証しない。

`FrozenEncoder.from_bundle()`は配布bundleの`best.pt`、`run.json`、
`feature-statistics.json`のみを必要とする。`run.json`に記録されたtrain manifestや
旧統計ファイルの絶対パスが利用できなくても、bundleだけで推論できる。
履歴を含むcheckpointはchecksumを確認してから読み、model stateだけを利用する。

## 登録入力と不足時の動作

入力JSONは`schema_version: 1`、`user_id`、`segments`を持つ。
`segments`の各要素は次のフィールドを持つ。

| フィールド | 意味 |
| --- | --- |
| `segment_id` | 入力内で一意な区間ID |
| `vowel` | `a` / `i` / `u` / `e` / `o` |
| `source_file` | `--audio-root`からの相対WAVパス |
| `start_frame` / `end_frame` | 元WAVの開始frameと終了frame（終了は含まない） |
| `source_sha256` | 任意。指定時は元WAVのSHA-256と照合する |

一つの入力には一人の登録音声だけを含める。入力が本当に本人の声であるかを自動で判断する
機能は含まない。元WAV、母音ラベル、frame境界は上流の抽出処理の成果物を利用する。

初期登録の最低区間数は**各母音10区間、全5母音必須**とする。これはPhase 3の主評価条件に
揃えるための初期値であり、必要数の最適性を意味しない。`--minimum-segments`で
最低数を明示変更できるが、母音の欠落を許容する部分プロファイルは作成しない。

- 最短30 ms未満、無音、raw RMSを小数点以下3桁へ丸めて-50 dBFS未満の区間は使用しない。
  RMSはcrop/DC除去前の元区間で計算し、Phase 2の規則に揃える。
- 除外した区間のID、母音、元WAV、frame範囲、理由を残す。
- 除外後に最低数を満たさなければ、母音別の利用可能数と不足内容を返す。プロファイルは保存しない。
- 未対応母音、不正なframe範囲、元音声範囲外、ファイル読込・形式・checksumの不一致、
  重複IDや同じ元区間の重複指定は登録全体をエラーにする。
- 音声root外の参照は許可しない。入力パスを正規化し、元音声checksumをプロファイルへ保存する。
- 全ての利用可能区間を等重みで使う。最低数を超えた区間も使用し、人手選別や品質重み付けは行わない。

## 代表値とPhase 3との関係

各区間のembeddingをCPUのfloat64へ変換してL2正規化し、母音ごとに算術平均した後、
平均ベクトルを再びL2正規化する。平均normが`1e-12`未満、または非有限なら登録を停止する。
集約方法はPhase 3の`phase3_train.evaluation._profiles`と同じである。

Phase 3は固定seedのsource round-robinで母音ごとに10区間を選び、1/5/10区間のprofileを
評価した。Phase 4は入力された全利用可能区間を`segment_id`順に集約するため、登録区間数は
可変である。同じ区間集合・同じembeddingなら同じ代表値となるが、加算順やbatchサイズの差で
浮動小数点の微小差が生じ得る。品質規則、中央crop、特徴統計、集約方法は維持する。

CLIの`prepare-jvs`は学習に使っていないvalidation話者に限定し、Phase 3と同じ選択処理で
各10区間を用意する。これは動作確認用であり、testを再評価したり閾値を再較正したりしない。

## プロファイル形式とPhase 5の境界

保存形式は一人につき一つのUTF-8 JSON。schema / registration / policyの版、ユーザーID、
集約方法、最低区間数、母音ごとの128次元unit vectorと使用区間数、入力区間と除外理由を保存する。
`encoder`にはseed、checkpoint・encoder設定・特徴統計・前処理設定・前処理/モデル実装の
SHA-256を持つ。末尾の`sha256`はそれ以外の全内容をcanonical JSONにしたchecksumである。
時刻や保存先パスを含めず、同じ入力・bundle・実行環境では同じファイルを再生成できる。

```python
from pathlib import Path
from registration import FrozenEncoder, load_profile

encoder = FrozenEncoder.from_bundle(Path("path/to/selected-bundle/20260926"))
profile = load_profile(Path("profiles/user.json"), encoder=encoder)
reference_a = profile.vector("a")  # numpy.float64[128]、L2 norm=1
# Phase 5は別の音声区間をencoder.embed(pcm, segment_id)へ渡し、同じ母音と比較する。
```

ライブラリを他の工程から利用するときは、このPhase 4ディレクトリをPythonのimport pathへ
追加する。`registration`がPhase 3の共有実装への参照を設定する。

`load_profile`は版、checksum、全5母音、vectorの次元・有限性・norm、区間数と由来の対応を
検査する。`encoder=`を渡すと生成時のモデル・前処理との互換性も検査する。
Phase 5は互換性確認を必須とし、同じ母音のvectorを利用する。
母音ごとの区間数は`profile.data["vowels"][vowel]["segment_count"]`で取得できる。
登録数が違うprofileにPhase 3の固定閾値を無条件で流用せず、照合条件・統合と較正はPhase 5以降で決める。

保存は一時ファイルから原子的に公開し、既存ファイルの上書きを拒否する。
このPoCではプロファイルの追記や認証成功後の更新APIを用意しない。

## 実行例

リポジトリrootから実行する。生成物はGit管理外の`artifacts/`へ置く。
2回目の登録確認は保存先を変える。各コマンドの`--help`で入力パスを変更できる。

```sh
uv run --project poc/phoneme-user-registration python \
  poc/phoneme-user-registration/scripts/register_user.py prepare-jvs \
  --speaker jvs007 \
  --output artifacts/phoneme-user-registration/smoke/jvs007-input.json

uv run --project poc/phoneme-user-registration python \
  poc/phoneme-user-registration/scripts/register_user.py register \
  --input artifacts/phoneme-user-registration/smoke/jvs007-input.json \
  --output artifacts/phoneme-user-registration/smoke/jvs007-profile.json

uv run --project poc/phoneme-user-registration python \
  poc/phoneme-user-registration/scripts/register_user.py inspect \
  --profile artifacts/phoneme-user-registration/smoke/jvs007-profile.json
```

## 検証

`tests/test_registration.py`は合成PCMと小型encoderを使い、JVSや学習成果物への依存なく、
保存・再読込、入力順序を変えた再現性、Phase 3との集約一致、parameter不変、最低数・欠落、
短区間・無音・低レベルの扱い、重複、不正入力、checksum・互換性・版・vectorの検査、
原子的保存と上書き拒否、trainデータなしのbundle読込を確認する。
2026-10-04に実際の選択bundleとvalidation話者`jvs007`でも登録・保存・再読込を確認した。
各母音10区間、合計50区間から生成した2回のプロファイルはファイル単位で一致し、
checkpoint・run・特徴統計のchecksumは登録後も採用時の値と一致した。
`artifacts/phoneme-user-registration/smoke/`に入力、プロファイルと
`verification.json`を保存した。Phase 3の35件とPhase 4の12件、合計47件のテストが通過した。
この確認は登録処理の動作検証であり、新しい認証性能の評価結果ではない。
