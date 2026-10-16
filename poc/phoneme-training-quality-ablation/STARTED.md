# 品質選別比較の本学習開始記録

品質監査・前処理はPR #90（https://github.com/muzin00/phonia/pull/90）でmainへマージ済み。
merge commit: `b462b62c1660fe1b9b99418d2d2460785c03f4d3`。
新しい学習実装は`feat/training-quality-ablation`で作成した。

本学習開始: 2026-10-16T02:01:39+09:00。
run: `artifacts/phoneme-training-quality-ablation/src490-quality-20261016-v1`。
3条件×3初期値、各音響216,000更新＋Transformer64,000更新。
元データ・品質選別・話者×音素の区間数を一致させたランダム除外を比較する。
全データ版で共通するTransformer登録485話者・問い合わせ22,931件を固定した。
ランダム対照の3抽選とも4,126区間を除き、品質除外との重なりは676／682／672区間。

単体テスト5件とRuff、全3条件のsmoke（400＋40更新）は完了。
smokeの固定142ファイル・完了6ファイルのhashを確認し、失敗マーカーなし。
smokeでtestを評価していない。本学習はsmokeの重みを引き継がず開始した。
開始後は凍結した実装・設定・入力を変更しない。

以前の1時間ごとの報告依頼を引き継ぐローカル監視を起動した。
監視状態: `artifacts/phoneme-training-quality-ablation-monitor/src490-quality-20261016-v1/monitor-state.json`。
初回の定期依頼予定: 2026-10-16T03:02:15+09:00。
完了・失敗・実行プロセス消失時にも報告を依頼し、送信成功後に監視を停止する。

開始時点で効能の結果は未確定。途中のselection EERを最終test結果として扱わない。
