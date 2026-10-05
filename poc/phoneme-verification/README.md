# Phase 5: 母音別の本人照合とスコア統合

[Issue #40](https://github.com/muzin00/phonia/issues/40)の照合処理。
Phase 4で保存したユーザープロファイルと、別の音声の同じ母音を比較し、
母音別スコアと5母音の統合スコアを算出する。

初期仕様と判断理由は[設計](design.md)を参照する。
JVSを使った[動作確認結果](smoke-results.md)も保存した。
コード・設定・テスト・Python依存関係はこのディレクトリに置く。
Phase 4の読込API・固定encoderと、Phase 3の前処理を参照する。

## 環境構築と検査

```sh
uv sync --project poc/phoneme-verification

uv run --project poc/phoneme-verification \
  python -m unittest discover -s poc/phoneme-verification/tests -v

uv run --project poc/phoneme-verification ruff check poc/phoneme-verification
uv run --project poc/phoneme-verification ruff format --check poc/phoneme-verification
```

Python 3.12を使用し、NumPy・PyTorchはPhase 4と同じ版に固定する。
ruffは開発用依存関係に含める。

## 入力と出力

照合入力は`schema_version: 1`、`query_id`、`segments`を持つJSON。
区間フィールドはPhase 4の入力と同じ`segment_id`、`vowel`、`source_file`、
`start_frame`、`end_frame`、任意の`source_sha256`を使う。
各母音1区間以上が必要。抽出済み母音を入力し、音声全体からのアライメントはPhase 1で行う。

```json
{
  "schema_version": 1,
  "query_id": "sample-001",
  "segments": [
    {
      "segment_id": "sample-001-a-01",
      "vowel": "a",
      "source_file": "audio/query.wav",
      "start_frame": 2400,
      "end_frame": 3600
    }
  ]
}
```

これはフィールド説明用の一部入力。実際には`i/u/e/o`もそろえる。
登録時と同じ元WAVを使う入力は拒否する。別名コピーもchecksumで検出する。

結果JSONの`vowels.a.score`等が母音別スコア、`verification_score`が統合スコア。
値域は`[-1, 1]`で、大きいほど似ている。本人である確率への変換や認証判定は行わない。
使用区間・除外理由・区間ごとのスコア、登録と照合の区間数、プロファイルのchecksum、
encoder・前処理の識別情報、規則と実行環境の版も記録する。
結果のchecksumと集計の整合性を再読込時に検査し、既存ファイルは上書きしない。

## 実行例

リポジトリrootから実行する。Phase 3の採用bundle、Phase 2のmanifestと元WAV、
Phase 4の保存済みプロファイルが必要。以下は既存の`jvs007`プロファイルを使う。
生成物はGit管理外の`artifacts/phoneme-verification/`へ置く。

```sh
uv run --project poc/phoneme-verification python \
  poc/phoneme-verification/scripts/verify_user.py prepare-jvs \
  --speaker jvs007 \
  --output artifacts/phoneme-verification/example/jvs007-query.json

uv run --project poc/phoneme-verification python \
  poc/phoneme-verification/scripts/verify_user.py verify \
  --profile artifacts/phoneme-user-registration/smoke/jvs007-profile.json \
  --input artifacts/phoneme-verification/example/jvs007-query.json \
  --output artifacts/phoneme-verification/example/jvs007-result.json

uv run --project poc/phoneme-verification python \
  poc/phoneme-verification/scripts/verify_user.py inspect \
  --result artifacts/phoneme-verification/example/jvs007-result.json
```

`prepare-jvs`はvalidationの`verification`または`cross_text_verification`のみを使用する。
各母音5区間を初期値とし、`--segments-per-vowel`で正の区間数を指定できる。
seed固定の元WAV round-robinで選び、スコアを見て選別しない。
`verify`は`--bundle`と`--audio-root`で配布bundleや元音声の位置を変更できる。
不足・不正入力は終了コード2を返し、結果ファイルを作成しない。

## 後続処理からの利用

このディレクトリをPythonのimport pathに追加する。
`verification`のimportがPhase 4・Phase 3の共有実装への参照を設定する。

```python
from pathlib import Path
from verification import load_verification_input, load_result, save_result, verify
from registration import FrozenEncoder, load_profile

encoder = FrozenEncoder.from_bundle(Path("path/to/selected-bundle/20260926"))
profile = load_profile(Path("profiles/user.json"), encoder=encoder)
query = load_verification_input(Path("queries/sample.json"))
result = verify(profile, query, encoder, audio_root=Path("path/to/audio-root"))
save_result(result, Path("results/sample.json"))
score = load_result(Path("results/sample.json"), encoder=encoder).score
```

## 実データの動作確認を再実行する

本人・別人・発話内容が異なる本人の3ケースを、validationだけで実行する。
各結果を入力順を逆にして再生成し、ファイル一致、登録と照合の音声分離、
モデル・特徴統計・プロファイルの不変を検査する。
照合スコアによる正誤判定や閾値調整は行わない。

```sh
uv run --project poc/phoneme-verification python \
  poc/phoneme-verification/scripts/run_smoke.py \
  --profile artifacts/phoneme-user-registration/smoke/jvs007-profile.json \
  --output-dir artifacts/phoneme-verification/smoke
```

再実行時は未使用の`--output-dir`を指定する。本人はプロファイルの`user_id`、
別人は初期値`jvs016`を使い、`--impostor-speaker`で変更できる。
