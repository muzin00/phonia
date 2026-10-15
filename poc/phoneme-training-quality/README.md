# 学習セットの品質検査と保留・除外

[全件調査](../phoneme-training-quality-audit/README.md)で見つかった不足に対し、
DC除去後の音量判定、録音単位のASR検査、音素区間単位の波形・長さ検査を実装した。
現行193万区間へ適用し、元区間との対応を保持した別版の学習候補を作成する。
既存の凍結済み抽出コード・入力・モデルは変更しない。

## 判定規則

規則は[`config.json`](config.json)に保存し、適用前に`design-freeze.json`で実装・入力とともに固定する。
閾値は先行するtrain品質調査に基づく初期規則で、EERで選んだものではない。

| 単位 | 条件 | 判定 |
| --- | --- | --- |
| 音素区間 | DC除去後RMS < −50dBFS | 除外 |
| 音素区間 | サンプルの1%以上が `abs(PCM / 32768) >= 0.999` | 保留 |
| 音素区間 | サンプルの過半数が完全なゼロ | 保留 |
| 音素区間 | 切り詰め前の音素長 > 1秒 | 保留 |
| JTube録音 | ASRの最小avg_logprob < −1 | 録音由来の全区間を保留 |
| JTube録音 | ASRの最大compression_ratio > 2.4 | 録音由来の全区間を保留 |
| JTube録音 | ASRテキストで同じ文字が20個以上連続 | 録音由来の全区間を保留 |
| JTube録音 | 必須のASR診断値がない・非有限・テキストが空 | 録音由来の全区間を保留 |
| 全区間 | 上記に該当しない | この規則による採用候補 |

除外と保留が重なる場合は除外とするが、該当理由はすべて保存する。
保留は誤認識・誤ラベル・不良音声と確認されたことを意味しない。
長母音や子音の閉鎖区間なども該当し得るため、保留データと元音声は残す。
ASR診断値は校正済み誤り確率ではなく、デコード窓に由来する指標である。

JVS・CV・SRC4VCはこのデータ作成経路ではASRを必須にしないため、ASR欠測を理由に保留しない。
これは提供テキストが正しいという判定ではない。
alignment score、DC除去前後のRMS、DC平均、PCM上下限への張り付き率も記録する。
alignment scoreの一律閾値は未校正のため採否に使わない。

## データと学習入力の整合性

`quality.py`は区間測定・録音のASR検査・採否判定を実装する。
`prepare.py`は全件監査のhash付き測定キャッシュを再利用する。
これは元manifest・特徴量・録音・alignmentとの対応を既に確認した測定値であり、
今回の開始時にキャッシュ、元学習manifest／特徴量、既存poolのhashと所有者を検証する。

DC除去は、中央切り出し済みの波形から平均を引いた後にRMSを求める。
監査値はfloat64で計算されているため、除外対象全件および閾値±0.01dBの区間について、
元WAVを再読込し、学習時と同じfloat32 Torchの平均減算・RMS計算でも合否の一致を確認する。
一致しなければ処理を失敗させる。

新しい`screened/`には採用候補だけを含め、以下を作り直す。

- 音響学習用の区間index、話者／音素group、特徴量、メタデータ。
- Transformer学習の登録音声・問い合わせ音声内の区間index。
- 五母音が揃わなくなった登録音声とその話者の問い合わせ、五母音不足の問い合わせ、
  有効な問い合わせがなくなった登録話者を取り除く。
- すべての参照の話者・音素・範囲、登録音声と問い合わせの録音非重複を再検査する。

既存の特徴量は既にDC除去を経ているため、残す区間の128次元特徴量は再計算せず複製する。
全行で元特徴量との完全一致を検証する。音量正規化の有効化や特徴標準化の再推定は行わない。
モデルに渡す既存の長さ・RMSメタデータも保持する。今回の変更は品質による選別に限定し、
DC除去後の品質情報は別manifestへ記録する。
これは既存表現との比較用候補であり、全前処理や特徴統計を刷新した学習セットではない。

## 2026-10-16の適用結果

run: `artifacts/phoneme-training-quality/src490-quality-20261016-v1`。
主な集計の保存版は[`summary.json`](summary.json)。

| コーパス | 採用候補 | 保留 | 除外 | 元区間数 |
| --- | ---: | ---: | ---: | ---: |
| Common Voice | 665,951 | 139 | 90 | 666,180 |
| JTube | 505,402 | 3,048 | 321 | 508,771 |
| JVS | 585,742 | 8 | 22 | 585,772 |
| SRC4VC | 169,372 | 498 | 0 | 169,870 |
| 合計 | **1,926,467** | **3,693** | **433** | **1,930,593** |

保留と除外の合計は4,126区間（0.214%）。この割合は不良率ではない。
理由別件数は重複を含み、低音量433、最大振幅近傍43、過半数ゼロ603、
長い元音素区間262、ASR低logprob由来2,484、ASR高compression由来324、
ASR文字反復由来3、ASR診断値欠測0区間。

### 利用可能データへの影響

- 音響学習の490話者・36音素は全て残る。全音素で正例ペアを作れる。
- 話者×音素groupは15,053→15,047。6groupが空になり、8groupが新たにsingletonとなった。
  singleton区間は680→685。詳細は`coverage.json`に保存している。
- Transformer登録話者は485→485。問い合わせは23,017→22,981で、五母音不足により36件減った。
  この欠損を無視して元の問い合わせを学習へ流すことはない。
- 登録・問い合わせの区間参照と録音の分離を全件検証済み。

### 検証

- 単体テスト12件: DCのみの波形、閾値境界、複数理由、ASR欠測・反復、非有限値、
  区間indexの再対応、登録／問い合わせの母音欠損、録音重複検出など。
- float32のDC除去を541区間で直接確認。合否不一致0、RMS差の最大約0.00000573dB。
- 採用候補の全1,926,467特徴行が元特徴量と完全一致。
- 既存の`Pool`・正例ペア抽選・`Bank`で読み込み、Transformerの485話者それぞれについて
  問い合わせ1件の特徴を未学習encoderで計算した。五母音mask・有限性・参照は正常。
  optimizer更新とEER算出は行っていない。
- Ruffチェック・formatチェックを実施。

## 実行

リポジトリrootから:

```sh
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality/prepare.py prepare
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality/prepare.py status
poc/phoneme-verification-evaluation/.venv/bin/python poc/phoneme-training-quality/verify.py
poc/phoneme-verification-evaluation/.venv/bin/python -m unittest discover -s poc/phoneme-training-quality/tests -v
poc/phoneme-verification-evaluation/.venv/bin/ruff check poc/phoneme-training-quality
```

完了済みrunへの`prepare`は入力・出力hashを再検証して終了する。
未完了の出力への追記や上書きは行わない。設定・実装・入力を変える場合は新しいrun名を使用する。
別設定は`prepare.py prepare --config /absolute/path/config.json`で指定できる。
`verify.py`はこのディレクトリの標準configを検証する。

## 成果物

- `design-freeze.json`, `complete.json`: 適用前の規則・実装・入力、完了後の出力hash。
- `source-quality.jsonl`: 録音ごとのASR診断値・テキスト・判定理由。
- `segment-quality.jsonl.gz`: 元の全区間のID・録音index・測定値・判定・全理由。
- `decisions.npz`: 元cache index順のstatus（0=採用候補、1=保留、2=除外）と理由bit。
  bitの定義は`quality.py`の`REASONS`。判定規則の版はdesign freezeに保存する。
- `screened/train-segments.jsonl`: 採用候補。新`cache_index`と元`source_cache_index`を保持。
- `screened/train-features.f32`, `pool.json`, `pool.npz`, `train-metadata.npy`, `train-inputs.json`:
  既存`Pool(screened, screened)`と`Bank`で利用できる音響・Transformer用の入力。
- `screened/source-segment-indices.npy`: 新index→元index。
- `dc-only-source-indices.npy`: DC除去後の音量だけで除外した場合に残る元index。
  別比較用の選択子であり、このファイルだけで独立した学習poolになるわけではない。
- `coverage.json`: 話者・音素の欠損とsingletonへの影響。
- `review-samples.json`: コーパス×音素×判定ごとに固定seedで1件、計279件。
  採用候補も含めることで見逃しを確認できる。全体から均等確率で選んだ標本ではなく、
  未聴取の診断用データである。人間の回答は`null`で、品質率として集計しない。
- `compatibility-check.json`: 既存の学習入力処理での確認結果。

## この段階で分からないこと

雑音・他話者の混入・誤テキスト・短い音素境界のずれは、今回の規則を通過する可能性がある。
保留データには正しい音声も混じり得る。候補と非候補双方を原音声と照合して、
規則の見逃しと過剰な保留を確認する必要がある。

EER改善の効果は未検証。次の比較では元データ・品質選別・同程度の量をランダムに除いた対照を
同じ初期値・更新数で比較し、データ量と品質の効果を分ける。
元の評価データは変更せず、既に観測したtestの結果は探索的評価として扱う。
