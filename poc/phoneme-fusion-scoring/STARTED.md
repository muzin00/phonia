# 実行開始記録

2026-10-17 00:31 JST、本番 `src490-fusion-scoring-20261017-v1` を開始した。
作業ブランチは `feat/phoneme-fusion-scoring-ablation`。

- 3方式×3初期値の9モデルを各64,000更新。音響encoderは保存済みの3モデルを凍結。
- 融合学習の登録485話者、query 22,931件、36音素。各モデル32 query×64,000更新で
  2,048,000 query提示、本人・他人合わせて4,096,000ペア提示を予定する。
- 各方式102,787パラメータ。3初期値すべてで初期stateのhashが方式間一致し、
  実データの初期スコア差も0。難例採点の独立計算との差は最大3.34e-16未満。
- trainと評価話者の非重複を確認。実装・設定・入力167ファイルをhash固定。
- 6単体テスト、Ruff check/format、実データ3方式×40更新のsmokeを完了。
  smokeでtestは推論せず、精度改善の根拠にしない。
- 本番の最初の初期値は3方式とも学習更新に到達。00:33 JST時点で失敗マーカーなし。

主比較はselectionで選定したdirect対weightedのtest平均EER。
weighted_no_klと固定64kモデルは診断・副評価。途中のselection EERを結論にしない。

run: `artifacts/phoneme-fusion-scoring/src490-fusion-scoring-20261017-v1`

以前の1時間ごとの進捗依頼を引き継ぐローカル監視:
`artifacts/phoneme-fusion-scoring-monitor/src490-fusion-scoring-20261017-v1`

監視は00:32:37 JSTに開始、最初の定期依頼予定は01:32:37 JST。
完了・失敗・プロセス消失時は報告依頼後に自動終了する。監視は学習を変更しない。
凍結済み実装と別の実行記録として、本ファイルを実行開始後に追加した。
