# Phase 3・10話者18設定の成立性確認

Issue #30の設計version 2.0.0、seed `20260926`。Apple M4 Mac mini（24 GB）で
CPU・float32・DataLoader 0 workerを使用した。実行前に
`config/execution-budget.json`へ1 runの上限を時間3,600秒、RSS 8 GiB、
成果物12 GiB、空き容量下限400 GiBとして固定した。

本比較16設定と限定比較2設定の全18 runが2,000 updateまで完走した。
全runで学習履歴の損失・gradient・learning rateが有限値で、学習時のembeddingも
有限値検査を通過した。各runの選択checkpointを再読込し、同じ固定validation trial
（各run 1,887,030件）でscoreと指標が完全一致した。最初のrunは250 updateで保存した
checkpointから再開した。失敗・資源上限除外・未着手は各0件。test splitは使用していない。

下表は探索計画の順序であり、性能順位ではない。`AAM+S`はAAM-Softmaxと母音内SupConの併用。
EERは10話者成立性確認の診断値であり、候補の採否・順位決定に使わない。

| 区分 | encoder | RMS | loss | checkpoint update | validation macro EER | 時間（分） | 最大観測RSS（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 本比較 | statistics_mlp | on | AAM | 250 | 17.00% | 7.4 | 1.55 |
| 本比較 | statistics_mlp | on | AAM+S | 500 | 16.61% | 7.3 | 1.23 |
| 本比較 | statistics_mlp | off | AAM | 250 | 17.20% | 7.1 | 1.12 |
| 本比較 | statistics_mlp | off | AAM+S | 500 | 16.22% | 7.2 | 1.17 |
| 本比較 | tdnn | on | AAM | 1250 | 21.76% | 9.4 | 1.44 |
| 本比較 | tdnn | on | AAM+S | 250 | 22.02% | 9.3 | 1.03 |
| 本比較 | tdnn | off | AAM | 1250 | 20.66% | 9.3 | 1.03 |
| 本比較 | tdnn | off | AAM+S | 1750 | 21.36% | 9.4 | 1.06 |
| 本比較 | waveform_cnn_k80 | on | AAM | 1750 | 25.03% | 18.8 | 1.02 |
| 本比較 | waveform_cnn_k80 | on | AAM+S | 1250 | 23.37% | 19.1 | 1.23 |
| 本比較 | waveform_cnn_k80 | off | AAM | 2000 | 24.21% | 18.3 | 1.04 |
| 本比較 | waveform_cnn_k80 | off | AAM+S | 1750 | 22.58% | 18.2 | 0.99 |
| 本比較 | waveform_cnn_k240 | on | AAM | 2000 | 30.30% | 18.6 | 1.18 |
| 本比較 | waveform_cnn_k240 | on | AAM+S | 1500 | 27.80% | 18.7 | 1.21 |
| 本比較 | waveform_cnn_k240 | off | AAM | 2000 | 29.89% | 18.6 | 1.27 |
| 本比較 | waveform_cnn_k240 | off | AAM+S | 1750 | 27.42% | 18.6 | 1.22 |
| 限定比較 | framewise_cnn | on | AAM | 1750 | 20.90% | 21.5 | 0.95 |
| 限定比較 | waveform_cnn_k240_context27 | on | AAM | 2000 | 31.54% | 23.3 | 1.26 |

合計実行時間は約4.34時間。固定trialの実体を各runへ追加保管した後の
最大run成果物は8.56 GiB、全体は約154 GiBで、各runの12 GiB上限内。
終了時の空き容量は616 GiB。全runのconfig ID・run ID・設定checksum、
個別結果、checkpoint、固定trial・scoreは
`artifacts/phoneme-speaker-encoder/comparisons/phase3-sanity-v2-10spk-seed20260926/`
に保存した（大容量のためGit追跡対象外）。比較対象18設定と実行条件は固定したまま、
次段階の70話者・3 seed比較へ渡す。
