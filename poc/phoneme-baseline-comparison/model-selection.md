# 発話全体speaker embeddingモデルの選定

## 1. 選定結果と確認範囲

- 選定日: 2026-10-06
- 対象: [Issue #45](https://github.com/muzin00/phonia/issues/45)の発話全体方式
- 採用方針: SpeechBrainの`speechbrain/spkrec-ecapa-voxceleb`を主比較に使用する。
- 根拠: 公開モデルカード、モデル設定、公式推論APIの確認。JVSでの性能は未測定。

今回は再現可能な既存方式を一つ選び、Phase 6の方式との比較を成立させることを優先する。
ECAPA-TDNNが最新モデルや日本語で最良のモデルであるとは主張しない。
実行環境とモデルファイルの版・checksumは[動作確認設定](config/embedding-smoke.json)と
`uv.lock`へ固定した。認証比較の条件は[比較設計](comparison-design.md)と
[比較protocol](config/comparison-protocol.json)へ記録した。

## 2. 選定基準

1. 提供元の公開済み重みを使い、発話から固定次元のspeaker embeddingを取得できる。
2. 学習データ、入力条件、前処理、照合方法を追跡できる。
3. 重みの利用条件が明示され、公式設定・推論処理を参照できる。
4. CPUでの実行を検証でき、既存のPython/PyTorch評価処理と接続できる。
5. JVSのtest性能を選定根拠にせず、実装の成立性をvalidationで確認できる。

これらは本プロジェクトの選定基準であり、各モデルのJVS性能を表すものではない。

## 3. 候補比較

| 候補 | 公開情報で確認した条件 | 今回の扱い |
| --- | --- | --- |
| SpeechBrain ECAPA-TDNN | VoxCeleb1+2の学習データ、16 kHz mono、192次元、80-bin filterbank、モデル表示ライセンスApache-2.0。[モデルカード](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)、[設定](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/main/hyperparams.yaml) | 主比較に採用する方針。公開重み・設定・embedding抽出・cosine照合を一つの実装で追跡できる。 |
| SpeechBrain x-vector | VoxCeleb1+2の学習データ、16 kHz mono、512次元、24-bin filterbank、モデル表示ライセンスApache-2.0。[モデルカード](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)、[設定](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb/blob/main/hyperparams.yaml) | 古典的なTDNNの参照候補として記録。初回の必須比較には追加しない。 |
| WeSpeaker ResNet34-LM | VoxCeleb2 Devの5,994話者、256次元のResNet34、large-margin fine-tuning済み、モデル表示ライセンスCC BY 4.0。公式一覧では16 kHzのONNX推論手順も公開。[モデルカード](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)、[公式一覧](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) | 別アーキテクチャの補助候補。追加する場合はtest測定前に版・前処理・照合設定を固定する。 |
| WeSpeaker ReDimNet2-B6-LM | 公式一覧にVoxCelebモデルとして掲載。ただし確認時点のモデルカード本文は空であり、学習・前処理・測定条件は追加調査が必要。[公式一覧](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)、[配布先](https://huggingface.co/Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM) | 今回は保留。配布の存在だけから主比較への適合を判断しない。 |

各ライセンス欄は配布元の表示を記録したものである。コードのライセンス、重みの表示、
学習データの利用条件は別に追跡する。

SpeechBrainのモデルカード上ではVoxCeleb1-test（Cleaned）のEERがECAPA 0.80%、
x-vector 3.2%と報告されている。この値は候補の背景情報であり、JVSでの予測性能でも、
Phase 6のEERとの直接比較でもない。ResNet34-LMにもAS-Normあり／なしの別条件があるため、
公開EERだけで候補を順位付けしない。

## 4. ECAPAを採用する理由

モデルカードに学習データ、入力条件、embedding抽出とcosine照合の使用方法がある。
設定から特徴抽出、発話内平均正規化、192次元の出力を確認できる。
既存のPhase 6もcosineに基づくため、発話embeddingから登録profileを作る処理と
同じtrialでの照合に接続しやすいと判断した。これは実装上の選定理由であり、性能の保証ではない。

x-vectorは参照候補として有用だが、初回比較の方式を増やす必要性は現時点では低い。
ResNet34-LMを追加すると一つのモデルだけに依存した結論を確認できる一方、
別の推論処理・前処理・重みの検証が必要になる。まずECAPAで共通評価を成立させる。
この判断はtestの結果に基づいていない。

## 5. 実装へ引き継ぐ方針

| 項目 | 方針 |
| --- | --- |
| モデル更新 | 公開重みを固定し、JVSでの追加学習・特徴統計更新は行わない。 |
| 元音声 | Phase 6と同じJVS元WAVを使用する。モデル入力は24 kHzから16 kHzへ変換し、変換の実装・パラメータを記録する。 |
| 照合入力 | 1元発話を1queryとし、主比較は発話全体、補助比較は共通中心の最大1・2・3・5秒からembeddingを生成する。VAD・無音除去は行わない。 |
| embedding | `encode_batch(..., normalize=False)`を使用する方針。ここでの`normalize`は保存されたembedding統計による正規化であり、後段のL2正規化とは異なる。 |
| 登録profile | 発話embeddingをL2正規化して平均し、平均ベクトルを再びL2正規化する方針。登録発話の選び方・音声量・重み付けは比較設計で定義する。 |
| 照合score | query embeddingと登録profileのcosine similarity。初回はPLDA、AS-Norm、学習によるscore融合を追加しない。 |
| 閾値 | 方式・登録数・照合長条件ごとにvalidationで目標FAR 1%の閾値を決め、対応するtestと異文へ固定適用する。 |
| 判定 | Phase 6と同じ`score >= threshold`を使う。モデルAPIの既定閾値・既定判定結果を認証指標へ流用しない。 |

embedding抽出の引数は[公式API](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.classifiers.html)で確認した。
[公式SpeakerRecognition実装](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py)は
`normalize=False`のembeddingをcosineで比較し、既定閾値0.25と`>`で判定する。
本評価ではembeddingとscoreの処理を明示し、閾値と判定規則を共通評価側で管理する。

## 6. 公平性と結果の解釈

母音方式はJVSのtrain 70話者で学習した採用encoderであり、ECAPAは外部のVoxCelebデータで
学習した公開重みである。学習データ規模・言語・収録環境・モデル構造・損失・入力帯域が異なる。
日本語JVSでの性能と、両者の学習話者の個人単位での重複有無は未確認である。
VoxCelebという別コーパス名だけから学習話者の重複がないと断定しない。

同じsplit・登録／照合発話・trial・目標FARで評価しても、差を音素分割だけの効果とは解釈できない。
今回確認するのは、固定した母音システムと公開済みの発話全体システムの性能差である。
音素分割の効果をさらに切り分けるには、同じ学習話者・学習条件で訓練した発話全体モデルなどの
対照が別途必要になる。

また、同じ登録元発話を使うだけでは、母音のみを使う方式と発話全体方式の利用音声量は揃わない。
Phase 6の各母音1・5・10区間に対応する登録元WAVを使い、主比較は同じ元発話、
補助比較は同じ上限長の照合波形を渡す。取得音声と実利用時間、文脈の重複、入力不足の分母を
[比較設計](comparison-design.md)へ定義し、方式内部の利用時間を強制的に一致させない。

JVSの統制収録、収録日・端末メタデータの不足、Phase 3からのtest観測済みという制約は
[Phase 6評価設計](../phoneme-verification-evaluation/evaluation-design.md)から引き継ぐ。

## 7. 動作確認と次の工程

2026-10-06に、専用環境、固定revisionの取得とchecksum検査、validationの5話者・11発話での
24→16 kHz変換、192次元embeddingの抽出、反復・別プロセス間の一致、CPU時間・メモリの
確認を完了した。[動作確認結果](smoke-results.md)と[再現手順](README.md)を参照する。
本書の`main`・`develop`リンクは調査用であり、実験の版固定には使用していない。

共通trial、登録音声量、前後各20 msの文脈方式、入力不足の扱いを[比較設計](comparison-design.md)へ定義した。
[短入力確認](short-input-results.md)で、音声量を揃える30 ms登録入力はpaddingエラー、70 msは成功となった。
推論の成功だけでは認証精度を判断できないため、元WAVごとの時間合わせ2条件を撤回した。
[改訂設計2.0.0](comparison-design.md)は同じ元発話の主比較と、最大1・2・3・5秒の照合波形長比較を行う。
登録profileを固定し、比較実装の検証とvalidationの閾値較正を完了して、新方式のtest測定前に実行計画を固定する。

動作確認で変更が必要になった場合は、理由と新しい版を記録して本書を更新する。
公開情報の確認と実行による確認は区別し、未測定の項目を成功済みと記載しない。
