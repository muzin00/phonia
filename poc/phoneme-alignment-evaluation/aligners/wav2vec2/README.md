# Wav2Vec2 phoneme CTC ONNX

`facebook/wav2vec2-xlsr-53-espeak-cv-ft`をFP32 ONNXへ変換し、Apple
Silicon CPU上のONNX Runtimeで実行する。

## 固定バージョン

- モデルリビジョン: `2c733782da5604684829819a5eb744c193fe9398`
- PyTorch重みSHA-256: `04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551`
- ONNX SHA-256: `fdfecb377f173a45d92502d698558521bb0107c118ef151d6896a2f76987c65a`
- ONNX形式: FP32
- ONNX Runtime: `1.30.0`
- Transformers: `4.57.6`
- PyTorch: `2.14.0`（変換と同等性検証だけで使用）

## 環境構築

リポジトリルートから次を実行する。

```sh
poc/phoneme-alignment-evaluation/aligners/wav2vec2/setup.sh
```

モデルはGit管理せず、既定では次へ保存する。

- 公式モデル: `${XDG_CACHE_HOME:-$HOME/.cache}/phonia/wav2vec2/source-<revision>`
- ONNXモデル: `${XDG_DATA_HOME:-$HOME/.local/share}/phonia/wav2vec2/<revision>/onnx`

セットアップは公式モデルを固定コミットから取得し、PyTorch重みと変換後ONNXの
SHA-256を検証する。

## 評価実行

```sh
cd poc/phoneme-alignment-evaluation
uv run --group wav2vec2 python scripts/validate_wav2vec2_onnx.py
uv run --group wav2vec2 python scripts/run_wav2vec2_alignment.py
uv run python scripts/normalize_wav2vec2_alignment.py
uv run python scripts/extract_vowel_segments.py \
  --input data/alignments/wav2vec2/normalized.jsonl \
  --output-directory data/segments/wav2vec2 \
  --manifest data/segments/wav2vec2/manifest.jsonl \
  --validation data/segments/wav2vec2/validation.json
uv run python scripts/compare_wav2vec2_julius.py
```

共通期待音素列をモデルのeSpeak IPA語彙へ明示的に写像する。CTCの最尤経路を
動的計画法で求め、隣接音素間のblankフレームは中点で分割する。期待列に`pau`が
ある場合は、そのblank区間を前後の音素へ含めずpauseとして保持する。

CTCモデルはblankを多く出力し、音素ラベルは短いピークになりやすい。このため、
この境界復元規則も評価対象の一部とし、得られた区間を正解境界とは扱わない。
