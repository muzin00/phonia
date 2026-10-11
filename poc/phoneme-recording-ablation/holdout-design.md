# 残響単独拡張：候補固定後の新規Common Voice評価

開発比較の事前スクリーニングを唯一通過した `single5`（合成残響のみ50%）を候補とする。モデルは既に完了した3 seedの最終checkpointをすべて固定し、同seedのclean追加学習3モデルと比較する。開発側では雑音EERの改善が小さく差の区間は0を含み、雑音FAR増加も認められた。今回はCommon Voice全体への有益性が再現するかだけを検証し、雑音頑健性や実運用採用の判断とは分ける。

既存学習70 client_idおよび前回評価60 client_idを除く、50発話以上の未使用client_idを固定hash順で60人選ぶ。新規30を校正、残り30をtestとし、各20発話を登録・30発話を照合する。新しい音声の加工・特徴抽出・採点前に、話者選択・モデル・コード・評価基準を凍結する。client_idと実人物の一致やJVSとの人物重複は保証できない。

元のG2P/Julius・QC・特徴統計・全36音素encoder構造・等重み統合・登録上限30区間を維持。登録と照合は同じモデルを使用する。圧縮音声とPCMで重複を確認し、前回holdoutを含む既存音声と一致するものや同一split内の重複は除外する。除外やalignment失敗のqueryは全入力FRRの拒否として残し、都合のよい別音声に入れ替えない。新しい閾値は各モデルの新規校正30話者だけでFAR1%に合わせ、test音声処理前に全閾値を固定する。

主評価は5母音必須、補助は4母音以上。3 seedの平均指標を比べる（ensembleや最良seed選択はしない）。FAR/FRRは採点不能を含む全入力、EERは採点可能な入力を対象とする。対応付き話者bootstrap2,000回で、候補−clean対照の差を計算する。**主評価でFRR差の95%区間上端が厳密に0未満、かつFAR差の上端が+0.1pp以下**の場合だけ、このコーパスにおける改善が支持されたと判断する。補助評価だけの改善で主評価の不合格を覆さない。学習seed一般の不確実性、別日・別端末、任意の雑音・残響には一般化しない。test後の閾値調整・seed選択・候補入れ替えは行わない。

```sh
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/holdout.py freeze
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/holdout.py validation
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/holdout.py test
OMP_NUM_THREADS=1 poc/phoneme-alignment-evaluation/.venv/bin/python poc/phoneme-recording-ablation/holdout_report.py
```
