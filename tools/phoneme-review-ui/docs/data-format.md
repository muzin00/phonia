# レビューデータ形式

## 方針

UIはアライナー固有の出力を直接読み込まず、共通のレビューデータを読み込む。PoC側の変換処理がMFAなどの出力からこの形式を生成する。

ブラウザへ渡すデータには実際の方式名を含めない。候補IDと方式名の対応表はPoC側で別に管理し、レビュー中の先入観と意図しない表示を防ぐ。

時刻はすべて元音声の先頭を基準とした秒で表す。レビュー専用に切り出したWAVは入力として要求せず、文脈付き再生と区間再生の範囲を元音声と設定値から決定する。

## 入力データ

完全な例は [`../public/examples/review-dataset.json`](../public/examples/review-dataset.json) を参照する。このファイルは開発サーバーから`/examples/review-dataset.json`として配信される。

### データセット

| フィールド | 内容 |
| --- | --- |
| `schemaVersion` | データ構造のバージョン。現在は`2` |
| `datasetId` | データセットを識別する安定したID |
| `datasetVersion` | 内容を更新したときに変えるバージョン |
| `title` | UIに表示する名称 |
| `protocol` | 評価手順のIDとバージョン |
| `playback` | 文脈付き再生で使用する前後の余白 |
| `form` | 区間ごとの設問と総合判定の設定 |
| `items` | レビュー項目の配列 |

### レビュー項目

`utterance.audioUrl`はUIの配信元から取得できるURLとする。ローカルファイルシステム上のパスはブラウザから直接読めないため、PoC側の元音声をサーバーのURLへ対応付ける。

`target.index`は発話内の0始まりの評価対象番号、`unitCount`は長母音表現を展開した期待母音数である。隣接区間を結合した数ではない。長母音や無声化候補など、表示に利用できる属性は`tags`へ入れる。

`target.textRange`は任意で、発話テキスト中の対応範囲をUnicodeコードポイントの0始まり半開区間`[start, end)`で表す。確実な対応を生成できない場合は省略する。

### 候補

候補は匿名の`id`と`status`を持つ。プロトコルv2では、各項目に利用可能な候補をちょうど1件だけ含める。

- `available`: `segment.startSec`と`segment.endSec`を持つ
- `missing`: 方式が対象区間を出力できなかったことを表し、推測区間は持たない

複数方式の比較候補はこのスキーマへ直接混在させない。方式間比較を実装するときは、原子区間との対応を明示した別の比較データを定義する。

## 回答データ

編集中の回答と現在位置は、`datasetId`と`datasetVersion`で分離したブラウザの`localStorage`へ自動保存する。JSONLへはユーザーが画面上部の保存操作を行ったときだけ、回答済みかつ未出力または変更済みの項目を`ReviewRecord`として追記する。同じ項目を再評価した場合も古い行は上書きせず、`revision`を増やして新しい行を追加する。

各回答には次を含める。

- 入力を特定する`datasetId`、`datasetVersion`、`itemId`
- 記録日時とリビジョン
- 候補ごとの設問回答
- 評価時点の候補区間スナップショット
- `accepted`、`rejected`、`uncertain`の総合判定またはスキップ理由
- 評価プロトコルとUIのバージョン

候補区間を回答にも保存することで、入力データが後から更新された場合でも評価時点の条件を追跡できる。

ファイル保存時、UIは対象回答を順番に`POST /api/reviews`へ送り、ローカルサーバーが`revision`と`recordedAt`を付与してJSONLへ追記する。`GET /api/reviews`は指定した`datasetId`と`datasetVersion`について、項目ごとの最新リビジョンを返す。読み込み時はJSONLの回答へブラウザ内の編集状態を重ねる。保存先は`PHONIA_REVIEW_OUTPUT`で変更できる。

TypeScript上の入力型は[`../src/domain/reviewDataset.ts`](../src/domain/reviewDataset.ts)、回答型は[`../src/domain/reviewRecord.ts`](../src/domain/reviewRecord.ts)に置く。入力データと回答データは、読み込み時と保存時にそれぞれ実行時検証を行う。
