# Julius Speech Segmentation Toolkit

Apple Silicon上でJuliusの音素単位forced alignmentを実行する環境を管理する。
Julius本体とsegmentation-kitはユーザー領域へ配置し、システム領域へインストールしない。

## 固定バージョン

- Julius: `4.6` (`3b7174d0d4091f5e6ebb917769822032d079996f`)
- segmentation-kit: `e0e8bbaf98e27d19dfc6fe8312be607ad03592ad`
- 音響モデル: `hmmdefs_monof_mix16_gid.binhmm`
- モデルSHA-256: `58952ccfe60f283c7efb1c22f9e012c7d297429700b7eeb4b60f7651a6e65940`

## 環境構築

Xcode Command Line Toolsを導入したApple Silicon Macで、リポジトリルートから実行する。

```shell
poc/phoneme-alignment-evaluation/aligners/julius/setup.sh
```

既定の配置先は次のとおり。

- ビルドソース: `${XDG_CACHE_HOME:-$HOME/.cache}/phonia/julius/source-v4.6`
- Julius実行ファイル: `${XDG_DATA_HOME:-$HOME/.local/share}/phonia/julius/4.6/bin/julius`
- segmentation-kitと音響モデル: `${XDG_DATA_HOME:-$HOME/.local/share}/phonia/julius/segmentation-kit`

Julius 4.6はApple Siliconの`__arm64__`判定とCoreAudio実装のヘッダーが不足するため、`patches/julius-4.6-apple-silicon.patch`を適用する。評価に不要な補助ツールはビルドせず、`libsent`、`libjulius`、`julius`だけをビルドする。

## アライメント実行

```shell
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/run_julius_alignment.py
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/normalize_julius_alignment.py
uv run --project poc/phoneme-alignment-evaluation \
  python poc/phoneme-alignment-evaluation/scripts/extract_vowel_segments.py \
  --input poc/phoneme-alignment-evaluation/data/alignments/julius/normalized.jsonl \
  --output-directory poc/phoneme-alignment-evaluation/data/segments/julius \
  --manifest poc/phoneme-alignment-evaluation/data/segments/julius/manifest.jsonl \
  --validation poc/phoneme-alignment-evaluation/data/segments/julius/validation.json
```

元の24 kHz WAVは変更せず、Julius入力だけを16 kHz・mono・16-bit PCMへ変換する。内蔵のひらがな変換は使用せず、`expected.jsonl`の基準音素列を直接与える。`pau`だけをJuliusの`sp`へ変換し、連続母音は結合しない。

上流の`segment_julius.pl`はJuliusの実行失敗を検出しないため使用しない。実行スクリプトは終了コード、forced alignmentの完了、入力と出力の音素列一致、時間区間を検査する。

Julius 4.6はDFA読み込み時に省略可能な`.dfa.forward`を探し、存在しない旨をログへ出すが、通常のDFAを読み込んでforced alignmentを完了する。正常終了の判定にはこのメッセージではなく、終了コード、完了マーカー、出力区間を使用する。
