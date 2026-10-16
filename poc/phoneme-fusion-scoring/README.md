# Phoneme fusion scoring comparison

音素区間・音響encoderを固定し、複数音素の統合関数を比較する。以前の重み集中の観察は
原因の証明ではない。本比較の陰性結果も、音素波形に含まれる識別情報の限界を証明しない。

## 事前固定した比較

最新の品質比較の `original` で学習済みの490話者encoderを3初期値とも再利用する。
音響モデルは一切更新せず、Transformerはすべて新規に学習する。

| 条件 | 最終スコア | 音素重みのKL正則化 |
| --- | --- | --- |
| weighted（主対照） | 従来の非負・総和1の重みによる音素類似度の加重平均 | 0.001 |
| weighted_no_kl（診断対照） | 同じ加重平均 | なし |
| direct（主介入） | 均等類似度平均＋文脈化した音素特徴から直接学習する残差 | なし |

`direct` は共通Transformerの各音素出力を同じ線形層でscalarへ変換し、有効音素について
平均した値を残差とする。残差にはsoftmax・clipを適用しない。スコアは音素類似度の
最小値〜最大値の範囲に制約されない。均等類似度平均は開始時の一致のための固定基準で、
独立に事前学習した追加モデルではない。返り値の均等係数は説明用重要度ではない。

3条件はTransformer構造・パラメータ数・初期state・初期スコア・BCE校正・AdamW・
dropout乱数系列・学習ペア・更新数を揃える。`weighted_no_kl` を置くことで、
主比較に含まれるKL除去の影響も確認する。最適化の難易度までは同一とは主張しない。
既存のMLP比較は音素ごとの重み生成器であり、この文脈付き直接採点方式とは異なる。

## データ・難例・予算

- 音素境界・区間・128次元の入力統計特徴・36音素・既存5母音gateは変更しない。
- 品質比較で使った共通の元データ側登録話者・query集合と、各seedの提示順を再利用する。
- 4,000更新までは元のランダム負例。その後は各32 queryのうち16スロットを乱数で選び、
  その中のCV queryだけを難例化する。train内の同一コーパスから、元負例を含む8候補を
  選び、凍結encoderの均等類似度が最高の他人を採る。本人・別コーパスは候補にしない。
- 難例の採点には本人側・他人側・queryの共通音素を使う。Transformerの出力・
  selection/testは難例選択に使わない。全条件に同一scheduleを渡す。
- 3初期値×3条件＝9モデル、各64,000更新。学習率は最初4,000更新が3e-4、以降3e-5。
  2,000更新ごとにselectionを評価。CPU 1 thread/worker、同時3 worker。

## 評価と解釈

主評価はselectionの3コーパス等重み平均EERが最小のモデル。同値は早い更新を採用し、
update 0も含める。最終64kモデルは事前固定した副評価で、主評価とのすり替えはしない。
全18モデルを固定し、全selection FAR1%閾値を保存してからtestを推論する。
主比較は `selected/direct_minus_weighted/macro`。コーパス別・3seed別・固定閾値FAR/FRRも
報告する。EER差の区間は話者・対応seedのbootstrap（1,000回）。採点対象は共通で、
未採点本人は全入力FRRに拒否として含める。独立実装のEER計算で照合する。

既存selection/testは観測済みで、新しい独立holdoutではない。encoderとTransformerの
seedは対応する3組で、完全交差ではない。副比較には多重比較補正をしない。
testを見てハイパーパラメータ・予算・採用条件を変えず、自動採用もしない。
過去の別プロトコルのEERを対照として混ぜない。

## 実行

```sh
poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-fusion-scoring/tests
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-scoring/study.py execute --smoke
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-scoring/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-scoring/study.py status
```

smokeは実データ・3条件×1seed・40更新で、test推論を行わず動作のみを確認する。
本番実装・設定・入力は `design-freeze.json` でhash固定する。途中再開は同一設計のみ。
状態・失敗・完了はrun直下の `state.json` / `failure.json` / `complete.json`、
結果は `results.json`。モデル、閾値、スコア、難例候補と独立採点監査を保存する。
