# Phase 6 評価設計 v1.0.0

## 1. 目的と現在の状態

Phase 4の登録プロファイルとPhase 5の照合・統合方式を固定し、学習外話者の本人認証性能を測る。
本書は評価前に定めた設計である。実装と測定後の性能は[評価結果](evaluation-results.md)へ記録する。
初期対象は既存JVSの通常発話。新しいモデル学習、スコア統合の学習、既存方式との比較は行わない。
既存方式との比較は、今回作成する同じtrialと音声条件を利用してPhase 7で行う。

## 2. 固定する方式

| 項目 | 固定条件 |
| --- | --- |
| encoder | Phase 3採用`statistics_mlp`、70話者学習、単一配布seed `20260926` |
| 登録 | Phase 4の`register_user`、母音ごとの正規化平均ベクトル |
| 照合 | Phase 5の`verify`、区間cosineの母音内平均、5母音の等重み平均 |
| 入力 | 24 kHz / mono / signed 16-bit PCM、元WAVのframe区間 |
| 前処理 | Phase 3の中央crop・DC除去・log-Mel・保存済みtrain特徴統計、RMS正規化なし |
| 品質規則 | 30 ms以上。無音・固定RMS規則のnear-silentを除外 |
| 実行 | CPU / float32 / batch=1、集計float64、勾配無効 |

モデル、特徴統計、プロファイルのオンライン更新は行わない。
複数seedのensembleや最良seedの再選択も行わない。
checkpoint・既存API・指標処理・入力manifestのSHA-256をprotocolに保持し、実行前に確認する。
共有実装に変更が必要なら、新しい設計版とvalidation検証を作成し、旧結果と区別する。

## 3. データ分割と今回の主張

正本はPhase 2の`config/dataset-split.json`。train 70、validation 15、test 15話者を維持する。
Phase 6でtrain音声を照合・較正に使用しない。

| データ用途 | 発話 | Phase 6での役割 |
| --- | --- | --- |
| enrollment | parallel100の001〜050 | 登録profile作成 |
| verification | parallel100の051〜100 | 主要評価、validationでは閾値較正 |
| cross_text_verification | nonpara30の全発話 | 異文条件の補助評価。閾値較正には使わない |

音声の候補母集団は`utterance-manifest.jsonl`、利用可能な母音区間は
`phase3-vowel-segments.jsonl`を使う。元発話を基準にすることで、母音が残らなかった発話も分母に含める。
話者・split・role・元WAV・checksumが一致することを照合し、登録と照合の元WAVを分離する。
同じWAVの別区間や別名コピーを登録・照合へ分けない。

JVSには収録日、個別セッション、マイクIDがない。ここで測れるのは統制収録内の別発話性能であり、
cross-textをcross-sessionと呼ばない。別日・別端末・雑音環境の外部評価は、
[既存の段階的検証戦略](../phoneme-speaker-dataset/validation-strategy.md)に従う追加工程である。
今回は新しい音声収集を実行の前提にしない。

Phase 3のtest話者と結果は既に観測済みである。今回のtestは固定方式の統合スコアを測る後続評価であり、
新しい未観測corpusによる独立検証とは扱わない。Phase 3結果から方式を再選択せず、
Phase 6のtestスコアを見た後の設計変更には別の未使用データが必要となる。

## 4. 登録条件

主条件は**各母音10区間**。補助条件として各母音1・5区間を同時に測定する。
Phase 3の`make_enrollment`をseed `20260930`、count=10で使用し、
元WAV round-robinで作成したrank 1〜10の同じ固定区間列から先頭1・5・10区間を取る。
区間長、品質属性、モデルのscoreで選別しない。

Phase 4のAPIへ各条件の区間集合を渡し、`minimum_segments`も1・5・10と明示する。
1・5区間はPhase 4の初期登録10区間から最低数を変更した評価条件であることを記録する。
平均・正規化方法は変更しない。Phase 4は区間ID順に集約するため、
Phase 3のrank順・batch推論との微小な丸め差を「同じファイル」とは主張しない。

全15話者・全5母音の登録10区間が作成できなければ、そのsplitの評価を停止する。
登録失敗した話者を除いて人数を減らす、既存profileを別の話者IDへ流用する、代替区間をscoreで選ぶ、
といった変更は行わない。

## 5. 照合入力とtrial

**1元WAV発話を1つのquery**とする。複数発話をまとめて不足母音を補わない。
母音ごとに、その発話内で機械規則を満たす全区間を使用する。照合区間数は発話によって可変であり、
区間の間引き、上限設定、人工的な同数化、モデルscoreによる選別は行わない。
区間ID順で計算し、母音別件数・元発話時間・利用区間の合計時間を保存する。

この単位を選ぶ理由は、Phase 5の動作確認のように複数発話をまとめる効果を主評価へ混ぜず、
実際の1回の音声入力に近い条件を作るためである。
validationの主要照合では735/750発話で全5母音があり、この条件で試行を構築できる。

各queryを同じsplitの全15話者の各profileと比較する。
同じ話者との1試行がgenuine、異なる14話者との14試行がimpostor。
全組み合わせを使用し、別人の選び方や難易度によるサンプリングは行わない。

```text
query_id = SHA256([split, role, utterance_id])
trial_id = SHA256([protocol_version, query_id, claimed_speaker_id, enrollment_count])
is_genuine = (query_speaker_id == claimed_speaker_id)
```

ID計算は既存`json_sha256`のcanonical JSONを使う。
正解話者・role・正解ラベルはPhase 6のtrial側に持ち、Phase 5のスコア計算には渡さない。
全登録数・全指標で同じquery集合と順序を使い、条件をscoreに合わせて差し替えない。

## 6. 入力不足と分母

全5母音のいずれかが不足した発話は`no_score`として記録する。
Phase 5の`IncompleteVerification`と同じ扱いとし、偽の数値や`-inf`をscore欄に入れない。
元発話は存在するが抽出済み区間がない場合も、理由付きの`no_score`として残す。
各queryの全15試行に同じスコア取得状態を対応させる。

不正なWAV、root外参照、checksum・版・split・frame範囲の不一致、登録との重複、
不正なembeddingはデータ/実装エラーとして評価を停止する。精度集計のために黙って除外しない。
`no_score`は認証として受け入れないものとし、次の両方を報告する。

| 指標 | 分母と計算 |
| --- | --- |
| score取得率 | scoreを得たquery数 / 全候補発話数 |
| 条件付きFAR | scoreありimpostorの誤受入れ数 / scoreありimpostor数 |
| 条件付きFRR | scoreありgenuineの誤拒否数 / scoreありgenuine数 |
| 全入力FAR | 誤受入れ数 / no_scoreも含む全impostor数 |
| 全入力FRR | (誤拒否数 + no_score genuine数) / 全genuine数 |

EER・ROC・DETはscoreありの試行だけで算出する。スコア取得率と全入力FRRを必ず並べ、
母音が不足した発話を外しただけで性能が良く見えることを防ぐ。
各split・role・話者で全入力数、scoreあり数、no_score数、理由、母音別件数を保存する。
15話者のいずれかが1件もscoreを得られない条件は「評価不能」とし、話者を削除して続行しない。

## 7. 閾値の決定と適用

受入れ規則は`score >= threshold`。閾値は**validation / verificationのscoreあり試行だけ**で決める。
登録数1・5・10それぞれについて、統合scoreと各母音scoreを別々に較正する。
母音ごとの閾値を統合scoreに使わず、testやcross-textで再較正しない。

主要な動作点はvalidationの他人受入率目標1%。補助動作点は0.1%とEER operating threshold。
Phase 3の`far_target_threshold`・`eer_operating_threshold`をそのまま利用する。

- FAR目標: validation impostor件数を`N`とし、許容誤受入れ数は`floor(target * N)`。
  境界のimpostor scoreにfloat64の`nextafter(..., +inf)`を適用し、同点を一括して拒否する。
  実測FARが目標以下となる最小の候補閾値を使う。少数試行で達成したFAR 0を保証とは解釈しない。
- EER動作点: 候補は`-inf`と全unique scoreの`nextafter(..., +inf)`。
  `abs(FAR-FRR)`最小、同点ならFAR最小、さらに同点なら閾値最小を選ぶ。
  曲線補間のEER値と、実際に適用できるこの閾値は区別する。

validationで選んだ閾値は、同じ登録数・同じscore種類のtest / verification、
validation / cross-text、test / cross-textへそのまま適用する。
各閾値に較正元、試行数、false accept/reject件数、実測FAR/FRRとchecksumを持たせる。
testで目標FARを超えても閾値を直さず、その実測値を報告する。

## 8. 主指標と補助指標

主条件は登録10区間、test / verificationの統合score。
主要な動作点の結果は、validationでFAR 1%を目標に固定した閾値での**条件付きFRR**とする。
必ず同じ閾値の実測FAR、件数、score取得率、全入力FRRを併記する。
事前の製品目標がないため、任意の精度を合否基準として設定しない。

併記する分離性能は統合scoreのpooled EER、ROC（FAR対TPR）、DET（FAR対FRR）。
各trialを1件として集計する。EERはscoreの同点を一括し、FAR-FRRのゼロ交点を線形補間する。
testのEER用曲線を算出しても、testから認証用閾値を作成したことにはしない。

補助結果は登録数1・5、FAR目標0.1%、EER動作点、cross-text、各母音の同じquery集合での平均score。
母音別EERとその単純平均も診断値として保存するが、統合scoreのEERとは別の指標名にする。
話者別の入力数、FRR、query側およびclaim側のFARも記録する。
genuineとimpostorの1:14という比率に依存するaccuracyは主指標にしない。

補助条件から最良の登録数・母音・動作点だけを抜き出さず、事前定義した全条件を報告する。
比較は方式の診断であり、Phase 3の区間単位EERや複数発話のsmoke scoreと同じ試行条件とは扱わない。

## 9. 不確かさ

test 15話者を単位とするspeaker bootstrapを10,000回、seed `20260929`、PCG64で行う。
話者IDは辞書順。15人を復元抽出し、異なる話者が2人未満のdrawは引き直す。
genuine試行の重みはquery話者の出現回数`n_i`、impostorは両端の`n_i * n_j`。
同じdrawを全登録数・role・score種類で共有する。

既存`weighted_bootstrap.py`の重み付きEERを再利用する。
固定閾値のFAR/FRRと全入力FAR/FRRも同じ重みで再集計し、
2.5/97.5 percentile（NumPy `method="linear"`）を95%区間とする。
queryごとのtrialを独立な標本と仮定した二項信頼区間は主結果に使わない。

閾値はbootstrap内でも固定する。この区間は評価話者のばらつきを表し、
学習・閾値較正・収録条件の不確かさをすべて含むものではない。
低FARの分解能`1 / impostor_count`と誤受入れ件数も併記する。

## 10. 実装と凍結の順序

1. protocolと入力checksumを検証する。原発話、利用区間、役割・話者・元WAVの整合性を調べる。
2. 合成fixtureでtrial、no_score、登録数の入れ子、計算式、同点、閾値適用、bootstrapをテストする。
3. validationの登録profile・全query・trial一覧を作成し、各登録数・各母音・統合scoreを計算する。
4. validation / verificationだけから閾値を保存し、cross-textには固定して適用する。
5. 完成したPhase 6コード、protocol、入力データ、validationのprofile・score・閾値、環境と、
   testの登録入力・query・trial一覧のchecksumを`evaluation-plan.json`へ保存する。
   testの入力一覧はscore非依存の同じ規則で生成し、test音声のencoder推論前に固定する。
6. testのprofileを生成し、凍結した条件でverificationとcross-textを一度の評価実行として計算する。
   testのprofile・score等の生成後のchecksumは実行manifestへ追記し、凍結済みplanを変更しない。
7. 指標・信頼区間・曲線・coverage・限界を保存し、モデル・統計・profileの不変を確認する。

実装やデータのエラーで中断した場合は、失敗記録を残す。
scoreを参照せず同一条件で再開できる範囲はresumeとして扱い、条件変更・bug修正は別runとする。
testを観測した後に修正版を同じデータで実行する場合、初回の評価と区別し、独立確認とは報告しない。

## 11. 成果物と実行時の確認

保存先はGit管理外の`artifacts/phoneme-verification-evaluation/<run-id>/`。
設定・コード・結果の要約はこのPhase 6ディレクトリへ置き、既存フェーズの成果物を上書きしない。

| 成果物 | 内容 |
| --- | --- |
| `evaluation-plan.json` | 凍結条件・環境・参照ファイルと実装のchecksum |
| `profiles/<split>/<count>/<speaker>.json` | Phase 4形式の固定登録profile |
| `queries/<split>.jsonl` | 全候補発話、区間と不足内容、query状態 |
| `trials/<split>.jsonl` | query/claim、登録数、正解ラベル、trial ID |
| `scores/<split>.jsonl` | 統合・母音別score、または理由付きno_score、参照checksum |
| `thresholds/validation.json` | 較正元と登録数・score種類ごとの固定閾値 |
| `metrics/<split>.json` | 指標、件数、coverage、条件付き・全入力の結果、95%区間 |
| `curves/` | ROC・DETデータと静的図 |
| `execution.json` | 実行/再開履歴、時間、入力とモデルの不変確認 |

Phase 5の結果JSONを各trialの計算根拠として再利用する。大きな重複provenanceを避けるため
一括scoreファイルへ格納する場合も、result checksumとquery・profileの対応を追跡できる形にする。
embeddingのcacheを使う実装では、Phase 4/5の公開APIで選んだtrialとの数値一致を先に検査し、
モデル・前処理・元WAV・frame範囲のhashをcache keyへ含める。

完了判断は、全条件が再現可能に実行され、閾値を変更せず指標と限界が報告されていること。
精度が低いことと、実装が設計どおり動かなかったことを分けて記録する。
