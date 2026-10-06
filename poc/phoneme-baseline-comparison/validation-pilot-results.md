# validation pilot結果（比較設計2.0.0）

後続の実行凍結・固定閾値test・全条件CIも完了した。[最終比較結果](evaluation-results.md)を参照。

2026-10-06に、境界内母音・前後各20 ms付き母音・ECAPAの3方式について、
共通query波形のfull・最大1・2・3・5秒、登録数1・5・10を実行した。
7話者・12照合発話から63個のprofileと3,780件の照合レコードを生成した。
境界内方式のprofile・full scoreはPhase 6と完全一致し、全長さ条件の公開API照合も通った。
このpilotは処理の成立性・追跡可能性・再現性の確認であり、認証精度の比較結果ではない。

## 条件と入力の選定

- [比較protocol](config/comparison-protocol.json): 2.0.0。
  SHA-256 `5169ea422bf53ca11d52b47591befabaed8def7ee2d238de3ef9a0b61167c4ba`。
- [pilot設定](config/validation-pilot.json)は推論前に固定した。
- 辞書順先頭3話者（jvs007・jvs016・jvs021）の各query roleの先頭発話を選び、
  role別の最短・最長と、1秒条件で5母音が揃う最初の発話を追加した。
  同順位はutterance ID順とし、scoreを選定に使わない。
- 追加発話の話者を含め、登録・claimed speakerはjvs007・016・021・025・045・068・070の7話者。
  通常発話6・別テキスト発話6。元発話長は2.272458〜18.613792秒。
- 1queryにつき本人1・他人6の照合を登録数別に実行し、Phase 6の252個のtrial IDを保持した。
  3方式×5長さ条件に展開して3,780件とし、欠測もレコードを残した。
- 登録はPhase 6のrank prefixを使用し、queryの長さ条件間で変更しない。
  ECAPAは選択anchorを含む登録WAVを重複除去し、source順に単位vectorを等重み平均後、再正規化した。
  7話者の登録数10までで233個の固有登録WAVを使用した。

母音推論はPhase 6のtorch 2.14.1、ECAPAはtorch/torchaudio 2.11.0・SpeechBrain 1.1.1。
両方Python 3.12.13・NumPy 2.5.3、CPU・float32・batch 1・各thread 1。
平均とcosineの計算はfloat64。モデル重み・学習時の特徴統計は更新しない。

## 確認結果

| 検査 | 結果 |
|---|---|
| 境界内の登録profile | 7話者×3登録数の21個すべてPhase 6とchecksum一致 |
| 境界内scoreとPhase 5公開API | 全長さ条件の1,260件でscore・no_scoreが一致 |
| cacheを使わない公開APIでの再計算 | 21個の登録profile、1,260件のscore・no_scoreすべて完全一致 |
| 境界内full scoreとPhase 6 | 252件の状態が一致、数値の最大絶対差0 |
| 母音2方式の固有embedding | 1,470個、128次元、有限・単位vector |
| ECAPAの固有embedding | 287個、192次元、有限・単位vector |
| 同一入力3回の推論 | 全embeddingでbitwise一致 |
| 別プロセス間の再現性 | 入力・embedding・profile・scoreの32ファイルでSHA-256一致 |
| 登録profileの固定 | 全方式・登録数で5長さ条件間のchecksum一致 |
| query母集団と照合ID | 全条件で同一query、score IDの重複・欠落なし |
| モデル・特徴統計・元WAV | 推論前後で不変 |
| 閾値校正・test推論 | 未実施 |

境界内方式は既存公開APIで元coreの形式・checksum・区間・品質と登録/query音声の非重複を検証する。
文脈方式も同じ元coreの検証を通し、その後で共通query区間内にclipして拡張する。
長い拡張区間には既存250 msのcenter cropを適用し、拡張後のRMSでanchorを再選別しない。
cacheはモデル情報・元音声checksum・実際の開始/終了frameで識別し、query境界が変わる入力を区別する。
同じ入力区間を共有する条件でもscore IDは長さ条件別に作る。
母音不足のqueryではembedding計算前に`no_score`を決める。
`inputs.json`の母音利用frame集計は利用可能なanchorのgeometryを表し、
実際に計算した入力は`embedding-inputs.jsonl`へ記録する。

通常発話と別テキスト発話でのscore生成可能なquery数は次のとおり。
母音2方式は同じanchor集合を使用するため、coverageは一致する。

| queryの最大長 | 母音方式：通常 | 母音方式：別テキスト | ECAPA：通常 | ECAPA：別テキスト |
|---|---:|---:|---:|---:|
| full | 6/6 | 5/6 | 6/6 | 6/6 |
| 1秒 | 1/6 | 1/6 | 6/6 | 6/6 |
| 2秒 | 3/6 | 4/6 | 6/6 | 6/6 |
| 3秒 | 4/6 | 5/6 | 6/6 | 6/6 |
| 5秒 | 6/6 | 5/6 | 6/6 | 6/6 |

数値scoreは2,940件、`no_score`は840件。
5母音のいずれかが不足すると母音2方式は数値を作らず、全入力の分母に残す。
ECAPAは母音抽出の成否に依存せず、同じ共通波形を入力する。
この選定は極端な長さと母音充足の例を意図的に含むため、上表はvalidation全体のcoverage推定には使わない。
全体のmetadata coverageは[改訂設計の入力確認](query-window-readiness.md)を参照する。
ECAPAで1秒推論が成功したことから、1秒の認証精度が十分とは判断しない。

初回worker所要時間は母音2方式約8.8秒、ECAPA約46.4秒。
モデル読み込み・音声読み込み・3回反復・profile・照合・検査を含むため、単発推論の速度指標にはしない。

## 保存と再現

初回runは`artifacts/phoneme-baseline-comparison/validation-pilot-20261006-v2`へ、
再実行は`validation-pilot-20261006-v2-recheck`へ上書きせず保存した。
入力選定・条件・区間・source checksumを`inputs.json`に、実装checksumを`preflight.json`に保持する。
方式別の`embedding-inputs.jsonl`と`embeddings.npy`、`profiles.jsonl`、`scores.jsonl`から、
元音声・実入力frame・model・profile・query・trial・長さ条件を追跡できる。

初回`pilot-report.json`のSHA-256は
`a8387e7813b4c5494cb904a89b4ba6aec66e50205cae4c34dd152ff853ef5c52`。
統合`scores.jsonl`は
`71f8bde0754667a03b18b51fbb3330892e95caf8b164668efc72d17eeef67f76`。
別プロセスの一致とcacheなしAPIの検査は再実行runの`validation-pilot-checks.json`へ保存した。
合計38テスト、Ruff lint・format検査を通過した。
再現手順は[README](README.md#validation-pilotを再現する)を参照する。

## 次の段階

全validationの15話者・1,200queryについて同じ処理を実行し、
通常発話のscored trialだけから方式・登録数・長さ条件・support別の閾値を決める。
母音不足はcoverageと全入力FRRへ反映し、話者/roleにscored queryがない条件を明示する。
validation結果・閾値・実装・resource planを固定した後、testへ進む。
pilotではFAR・FRR・EERを比較せず、閾値・方式・長さの優劣を選定しない。
