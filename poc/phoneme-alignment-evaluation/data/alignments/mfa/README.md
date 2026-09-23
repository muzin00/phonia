# MFA alignment results

Montreal Forced Alignerによるモデル固有出力と、共通母音区間へ変換した結果を保存する。

## 生成方法

リポジトリルートから次を実行する。

```shell
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda env update \
  --name phonia-mfa \
  --file poc/phoneme-alignment-evaluation/aligners/mfa/environment.yml \
  --prune
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/run_mfa_alignment.py
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/normalize_mfa_alignment.py
```

## 成果物

- `input/`: manifestから生成するMFA用テキスト。Git管理しない
- `raw/`: MFAが出力したモデル固有の単語・音素境界JSON
- `normalized.jsonl`: MFAが出力した母音phone区間を結合せず共通表記へ変換した結果
- `run.json`: 実行環境、モデル、処理時間などの記録
- `validation.json`: 期待母音列との一致と時間区間の機械検査結果

正規化時には`expected-vowels.jsonl`と母音列・構成単位数を照合する。不一致がある場合は成果物を正常終了として扱わない。

Apple SiliconではMFAの並列特徴量生成がsegmentation faultになる場合があるため、スクリプトは既定でmultiprocessingを無効にする。調査目的で有効化する場合だけ`--use-multiprocessing`を指定する。
