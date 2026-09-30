# Phase 3・70話者30分上限pilot

Issue #32の正式な54 runに先立つ時間・容量計測。Apple M4 Mac mini（24 GB）のCPUで、
seed `20260926`、70話者、float32、DataLoader 0 workerを使用した。
6設定を実行し、最大4 processまで一部並列にした。各processには起動から1,800秒の
強制上限を設けたが、全て上限に達せず正常終了した。

学習条件は正式設計の最大30,000 update、warmup 1,000 update、validation間隔
1,000 updateを維持し、`--target-update 1000`で早期に区切った。固定validation
trial全1,887,030件を評価した。これは学習のearly stoppingでも、正式な30,000 update
比較でもない。test splitは使用していない。

| encoder | 完了update | 学習＋validation時間 | validation macro EER | trial数 | 成果物容量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `waveform_cnn_k240_context27`（RMS on・AAM） | 1,000 | 488.6秒（8分09秒） | 31.39% | 1,887,030 | 約1.0 GiB |
| `statistics_mlp`（RMS on・AAM） | 1,000 | 131.2秒（2分11秒） | 12.40% | 1,887,030 | 約1.0 GiB |
| `tdnn`（RMS on・AAM） | 1,000 | 257.5秒（4分18秒） | 14.96% | 1,887,030 | 約1.0 GiB |
| `waveform_cnn_k80`（RMS on・AAM） | 1,000 | 670.7秒（11分11秒） | 29.94% | 1,887,030 | 約1.0 GiB |
| `waveform_cnn_k240`（RMS on・AAM） | 1,000 | 670.8秒（11分11秒） | 32.29% | 1,887,030 | 約1.0 GiB |
| `framewise_cnn`（RMS on・AAM） | 1,000 | 492.0秒（8分12秒） | 14.10% | 1,887,030 | 約1.0 GiB |

全runでupdate 1〜1,000の履歴が連続し、loss、gradient norm、learning rate、
accuracyは全件有限値だった。validationは`partial: false`、splitは`validation`。
各runのcheckpoint、履歴、score、metricsは次のGit管理外の場所に保存した。

- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-context27-seed20260926/`
- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-statistics-mlp-seed20260926/`
- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-tdnn-seed20260926/`
- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-waveform-cnn-k80-seed20260926/`
- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-waveform-cnn-k240-seed20260926/`
- `artifacts/phoneme-speaker-encoder/comparisons/phase3-70spk-30min-pilot-framewise-cnn-seed20260926/`

同機は物理10コア（性能4＋効率6）、PyTorchの既定はprocessあたり4計算スレッド。
観測時のCPU使用量は3本の学習で合計約6コア相当、4本で約8コア相当、3本の同時validationで
約9コア相当だった。4本同時の学習は成立したが、同時validationはCPU競合が強い。
暫定的には**同時3 runを基準、4 runを上限**とし、評価開始をずらす運用を検討する。
同じ設定で並列数を変えた比較はまだ行っておらず、これを最適値とは主張しない。

この6値は構成の採否・順位決定に使わない。とくに重い設定は1,000 updateだけで
約8分かかり、正式な30,000 updateを1 run 30分以内に収める見込みはない。
30分上限を正式比較へ適用するなら、全18設定・3 seedに共通する学習・選定規則を
validationの性能を見る前に別版として事前固定する必要がある。
