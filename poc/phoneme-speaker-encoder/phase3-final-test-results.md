# Phase 3・採用encoderの最終test評価

Phase 3設計version 2.0.0の最終評価。採用済みの
`log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel`を、
70話者で学習した3 checkpointと各seed自身のvalidation閾値のまま、未知のtest話者15人で一度評価した。
設定・checkpoint・閾値・手順はtest selection生成前に`plan.json`へ固定した。
test結果による構成・seed・閾値の再選択、再較正、ensembleはしていない。

## 主結果

主条件はverification、登録10区間、5母音EERの等重み平均。
EERはtest scoreから得た分離性能で、FAR/FRRはvalidationで固定した閾値のtest実測値。
後者は母音とseedを等重み平均した値。

| seed | test macro EER | cross-text EER | 登録1区間EER | 登録5区間EER | FAR 1%用閾値のtest FAR | 同FRR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260926 | 10.017% | 10.548% | 19.673% | 11.726% | 1.185% | 43.680% |
| 20260927 | 10.063% | 10.669% | 19.285% | 11.428% | 0.882% | 45.245% |
| 20260928 | 10.004% | 10.641% | 19.480% | 11.422% | 1.136% | 40.956% |
| 3 seed平均 | **10.028%** | **10.619%** | **19.480%** | **11.525%** | **1.068%** | **43.294%** |

主EERのseed間母標準偏差は0.025 percentage point。
validationの9.436%よりtestは0.592ポイント高かった。
FAR 0.1%用のvalidation閾値では、testで平均FAR **0.122%**、FRR **71.050%**。
FAR目標の達成はtestや実環境で保証されない。特に低FARでの棄却率が高く、
この結果だけで実際の認証用途に十分な性能と判断できない。

| 母音 | test EER・3 seed平均 |
| --- | ---: |
| /a/ | 8.742% |
| /i/ | 11.873% |
| /u/ | 13.378% |
| /e/ | 6.831% |
| /o/ | 9.316% |

登録1・5・10区間の3 seed平均macro EERはそれぞれ19.480%、11.525%、10.028%。
上表のseed別登録数EERと、母音別の固定閾値FAR/FRRは成果物JSONを正本とする。

## 事前定義された品質層

主条件の長さ別macro EERは30–49 ms **12.820%**、50–99 ms **9.096%**、
100 ms以上 **8.011%**。無声化なしは9.567%。無声化ありは/i/・/u/だけに存在し、
それぞれ281/288 query、EERは35.469%/35.822%だった。
他の3母音に該当queryがないため、無声化ありの5母音macro EERは定義しない。
品質フラグは全queryで空、長母音属性は全件falseで、これらの属性間比較はできない。
250 ms超の区間は母音別に3～18 queryしかなく、層別EERを安定した推定とは扱わない。

## 手順・成果物

testの固定登録区間は750件、verificationとcross-textを含む照合queryは41,831件。
各seedで42,581 embeddingを計算し、固定trial **1,882,395件**を採点した。
3 seedのembedding計算・音声データ読込・採点・集計の実測時間は
18.66 / 17.78 / 17.92秒。モデル初期化・checkpoint読込と選択ファイル生成は含まない。

成果物ルート（Git管理外）:
`artifacts/phoneme-speaker-encoder/final-test-phase3-selected-v2/`。
`plan.json`に採用記録、3 checkpoint・seed別閾値・train特徴統計・評価コードのSHA-256、
単一配布用seed `20260926`、評価条件を記録した。
`fixed-test/data.json`にtest登録・trialのchecksumを、各`<seed>/scores/test.jsonl`と
`<seed>/metrics/test.json`に全score、EER/ROC/DET、固定閾値のFAR/FRR、品質層を保存した。
全3 seedのscore checksumは集計時に再照合した。

| 成果物 | SHA-256 |
| --- | --- |
| `plan.json` | `1685e1881173f9bca9d97265c4ff8c357e97245bb8bdb7109f452bf3b20be101` |
| `fixed-test/data.json` | `7ea8266bb1c20b16851f56dbe6f0647727925c9f54ec0adfe6e4530dab9867f9` |
| `results.json` | `0941050dfab9fc068996056ad81b2cdc371363315f2a56db1d4130cffe8ca955` |

評価コードの事前コミットは`4bb18c7`。計画の`git_commit`・実装checksumと
[70話者比較結果](phase3-70spk-comparison-results.md)の採用checkpoint checksumが一致する。
保存済みの3 seedから集計を再検証するコマンドは次の通り。
評価本体は事前コミット`4bb18c7`とその実装checksumに固定された一度限りの実行である。

```sh
poc/phoneme-speaker-encoder/.venv/bin/python \
  poc/phoneme-speaker-encoder/scripts/build_final_test_report.py
```

testは候補18設定とcheckpointをvalidationで選んだ後の、別話者15人での一度の最終確認である。
3 seedは同じtest話者を共有し、seed間標準偏差は評価話者母集団の信頼区間ではない。
今回の相対的な分離性能と固定閾値の結果は、収録機器・雑音・実際の登録フローを含む
運用環境での認証性能を保証しない。運用可否は用途別のFAR/FRR要件と外部評価で判断する。
