# 品質選別後の比較学習

品質検査v1を適用した学習入力について、元データと区間数を揃えたランダム除外対照から
音響encoder・Transformerをそれぞれ再学習する。品質検査の保留は不良確定を意味しない。

## 固定する比較

| 条件 | 音響学習区間 | 選択方法 |
| --- | ---: | --- |
| original | 1,930,593 | 元の採用済み区間すべて |
| screened | 1,926,467 | 品質v1の保留3,693・除外433区間を外す |
| random_matched | 1,926,467 | 話者×音素ごとの残存数をscreenedと完全一致させ、元の各group内から無作為に残す |

モデル初期値20261101・20261102・20261103の各3回、計9モデルを新規初期化する。
random_matchedのデータ抽選は対応する20261111・20261112・20261113の3回で、
モデル初期値ごとに別の対照を用いる。音響216,000更新＋Transformer64,000更新／モデル。
旧モデル、旧test結果、RL方策は再利用しない。方式を変えるための途中打ち切りはしない。

全条件で[前回CV比較](../phoneme-common-voice-ablation/README.md)の一様な音素・話者抽選と
singleton補助分類を再利用する。screenedとrandom_matchedは話者×音素数が同じなので、
空group・singletonの構成も一致する。従来の音響loss・学習率・特徴表現を保持する。

### Transformerの比較条件

データを除くと問い合わせの五母音充足が変わるため、学習前に、元データ・screened・
3つのrandom_matchedの**全5データ版で使用可能な登録話者と問い合わせIDの共通部分**を取る。
各条件の登録／問い合わせの中身はその条件で残った区間だけを使う。
共通話者・共通問い合わせを同じ順序に並べ、初期値ごとに同一の問い合わせ・負例scheduleを使用する。
負例は問い合わせと同一コーパス内の別話者から一様抽選する。

このためoriginalは音響入力を全件使うが、Transformerの問い合わせ集合は共通化した新しい対照である。
前回CVありモデルとの比較を主結果にしない。

## 検証と評価

準備時に既存データのhashを検証し、各ランダム対照の話者×音素数の一致、
特徴全行の元入力との一致、登録／問い合わせの所有者・音素・録音分離を確認する。
区間選別・問い合わせ共通化にselection/testのスコアを使わない。

全条件同じselection/testを利用する。Transformerは更新0を含む2,000更新ごとのselection平均EERで、
最初の最小値のcheckpointを選ぶ。全9モデルを固定し、selection FAR1%の閾値も固定した後でtestを評価する。
各コーパスと等重み平均EER、FAR/FRR、採点率、区間露出・学習コストを保存する。
screened−original、random_matched−original、screened−random_matchedの差と、
評価話者・対応する学習seedを再標本化した95%区間を出す。複数比較補正はしない。

以前から観測済みの評価セットであり、結果は探索的な比較。
3つのランダムデータ抽選と3つのモデル初期値を対応させるため、両者の分散は分離できない。
録音ID、音素内の音響内容、登録／問い合わせ内の区間数までは一致しない。
改善しても、特定の音声が不良だったことや個別品質指標の因果効果は証明しない。
元モデルの自動置換は行わない。

## 実行

```sh
poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-training-quality-ablation/tests -v
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality-ablation/study.py execute --smoke
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality-ablation/study.py execute
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality-ablation/study.py status
```

smokeは全データ版と3条件を読み、1モデル初期値・音響400更新＋Transformer40更新で動作確認する。
testへアクセスせず、効能の結論を出さない。本比較とは別runで、smokeの重みを本比較へ引き継がない。
本比較は3workerで実行し、同一runの同時起動をlockで防ぐ。
失敗後は凍結済み設定・コードを保ったまま`execute`でcheckpointから再開できる。

run: `artifacts/phoneme-training-quality-ablation/src490-quality-20261016-v1`。
準備済みデータ、schedule、比較の監査結果は`prepared/`。
学習開始前の実装・設定・入力hashは`design-freeze.json`、各条件・初期値の進捗は
`<arm>/seed-<seed>/state.json`と`acoustic/state.json`・`fusion/state.json`。
完了結果は`results.json`・`complete.json`で確認する。実行開始後は凍結済み実装を変更しない。
