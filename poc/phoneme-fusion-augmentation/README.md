# 音素統合学習へのノイズ・欠損追加

全36音素encoderを固定した[前回のTransformer検証](../phoneme-transformer-fusion/README.md)では、cleanの平均EERは改善したが、照合区間の一部にノイズを加えると等重みより悪化した。今回は統合モデルのtrain入力にノイズ・子音欠損を追加し、clean性能と欠損・ノイズ耐性を測る。

結果は[HTML比較表](evaluation-results.html)、[全seedと学習曲線](evaluation-results.md)、[数値JSON](evaluation-results.json)へ出力する。

## 固定する比較条件

- 全36音素encoder、登録上限30区間/音素、JVS・Common Voiceの同じ140 train話者。SRC4VCは使わない。
- 前回と同じMLP・Transformer構造、seed 20261015/20261016/20261017、初期状態、本人/同一corpus他人ペア列、各4,000更新。encoderは更新しない。
- 変更はtrain入力加工のみ。clean / noise / missing / noise+missingを各25%で提示する。両方式で同じ加工スケジュールを使用する。
- noiseはquery側の実波形区間を確率20%で選び、SNR 10dB白色ノイズを加えて固定encoderで再抽出する。各queryは固定の1変種。train seed 20261019はtest seed 20261018と別。
- missingは子音を確率50%で欠損させる。5母音は残す。登録側を加工せず、本人/他人ペアで同じquery加工・音素maskを共有する。
- 採用は前回と同じclean validation通常EER最小、同値は早い更新。update 0も候補。各モデルのclean validation通常スコアからFAR 1%・FAR 0.1%・EER動作点の閾値を決める。
- 全6モデル・閾値のSHAを固定してから、前回と同じclean・子音50%欠損・20%区間SNR 10dBノイズのtestを採点する。testの音声加工と音素別cosineは前回と同じ。
- 等重み、前回clean学習6モデル、追加学習6モデルを並べ、seed最良値で方式を選び直さない。平均EERはensembleではない。

## 再実行

リポジトリrootから実行する。前回の固定済みartifactと元音声を必要とする。`prepare`は未使用の出力先のみ許可する。configに指定した実測runを上書きしない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-augmentation/experiment.py prepare --run artifacts/phoneme-fusion-augmentation/replay
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-augmentation/experiment.py train --run artifacts/phoneme-fusion-augmentation/replay
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-augmentation/experiment.py evaluate --run artifacts/phoneme-fusion-augmentation/replay
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-augmentation/auditing.py --run artifacts/phoneme-fusion-augmentation/replay
```

既定runの表を生成する:

```sh
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-fusion-augmentation/render_results.py
OMP_NUM_THREADS=1 poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-fusion-augmentation/tests -v
```

`design-freeze.json`は入力・加工・コード・旧artifactを学習前に固定し、`selection-freeze.json`は選択済みモデルと閾値をtest前に固定する。独立監査は音素別cosineから重み付きscoreを再構成し、別実装でEER・固定閾値FAR/FRRを計算する。元音声hash・train/evaluation非重複・登録側不変・初期状態・ペア列・4,000更新・採用規則も確認する。

同じ15 test話者の2,000回bootstrapで、追加学習Transformer平均と前回Transformer平均・等重みとの差を測る。各drawの話者重みを本人pairと他人pairへ対応させ、別実装で一部を照合する。HTML・Markdown・JSONの全表セルとリンクを検証する。

## 解釈の範囲

testは前回参照済みで、今回の追加検証も同じ集合を使う。bootstrapはこの固定モデル群に対するtest話者不確実性で、学習seed全般の信頼区間ではない。ノイズと欠損をまとめて追加しているため、それぞれの寄与は分離していない。音素境界とQCは元のものを固定し、加工後の再alignmentは行わない。
