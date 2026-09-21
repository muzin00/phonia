# JVS pilot samples

音素アライメント3方式の初期スモークテストに使用するパイロット音声です。

## Source

- Corpus: [JVS (Japanese versatile speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- Subset: `voiceactress100`
- Utterance: `VOICEACTRESS100_001`
- Downloaded: 2026-09-21

公式配布ページで公開されている、同一文章を3話者が読み上げたサンプルを使用しています。

| File | Speaker | Duration |
| --- | --- | ---: |
| `jvs001_VOICEACTRESS100_001.wav` | `jvs001` | 8.621042 s |
| `jvs002_VOICEACTRESS100_001.wav` | `jvs002` | 7.509333 s |
| `jvs003_VOICEACTRESS100_001.wav` | `jvs003` | 8.910083 s |

すべて24 kHz、16-bit PCM、モノラルのWAVです。

## Transcript

> また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。

## Usage notes

JVS公式配布ページに記載された利用条件が適用されます。音声データは、学術研究、非商用研究、個人利用など、同ページが認める範囲で使用してください。商用利用や再配布についても、利用前に最新の条件を確認してください。

付属または第三者提供の自動音素アライメントは、3方式の比較における正解データとしては使用しません。本プロジェクトでは、構造・信号の自動検査、方式間の境界一致度、候補区間を匿名化した支援レビューによって抽出品質を評価します。

機械可読なメタデータとSHA-256チェックサムは [`manifest.jsonl`](manifest.jsonl) に記録しています。
