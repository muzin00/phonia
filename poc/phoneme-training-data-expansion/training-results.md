# Phase 8: 学習データ追加の学習結果

追加学習1 runが完了。条件は2つ、seedは20260926の1つ。追加70ラベル・8,283発話から198,446母音区間を採用した。validation macro EERは9.196→9.274%。

|条件|話者ラベル数|学習区間数|encoder parameters|学習update|採用update|validation macro EER|
|---|---|---|---|---|---|---|
|JVSのみ（既存run）|70|323,913|65,920|18000|15000|9.196%|
|JVS＋Common Voice（新規run）|140|522,359|65,920|18000|15000|9.274%|

このEERは母音単一区間の照合を5母音で平均した学習時の指標で、checkpoint選択に使ったvalidationの値。発話単位のFRR/FAR、母音不足対応との組み合わせ、独立holdoutは未評価。追加データで話者ラベル数・区間数・収録環境が同時に変わるため、純粋なデータ量だけの効果ではない。

JVSのtrain/validation/testラベルを分離し、追加データは配布元のtrainだけを使用。匿名client_idとJVS話者の実在人物の重複は確認できない。人物の再特定は行っていない。1 seed・既存validationによる探索的な結果で、未知話者一般への改善は保証しない。testは今回の学習・選択・追加評価で使用していない。

採用重み: `artifacts/phoneme-training-data-expansion/training-data-20261008-v1/expanded/bundle/encoder.pt`。特徴統計: 同bundleの`feature-statistics.json`。再読込した重みの128次元embeddingはvalidation 8区間でbit一致。encoderは65,920 parameters、追加条件の140ラベル分類headは17,920 parametersで、推論用bundleには含めない。

追加データの整列成功8,280/8,283発話、失敗3発話。除外9,623区間。元音声は11.509時間（失敗発話も含む）、採用母音の合計は3.721時間（250 ms crop前）。

[固定条件・再現手順](README.md)、[HTML表](training-results.html)、[全数値CSV](training-comparison.csv)、[結果JSON](training-results.json)。
