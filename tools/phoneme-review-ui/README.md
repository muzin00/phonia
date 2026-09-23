# Phoneme Review UI

音声波形とモデルが推定した区間を視覚・聴覚で評価するためのローカルWeb UIです。

## 必要な環境

- Node.js 24.21.0
- pnpm 12.5.1

このディレクトリの `.node-version` にNode.jsのバージョンを記録しています。fnmを使用する場合は次のように準備します。

```sh
eval "$(fnm env --shell zsh)"
fnm use
```

## セットアップ

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Viteの開発サーバーが表示するURLをブラウザで開きます。

開発サーバーは、既定ではPoCの`poc/phoneme-alignment-evaluation/data/samples/`を`/media/`として配信します。別の音声ディレクトリを使用する場合は、起動時に指定します。

```sh
PHONIA_REVIEW_MEDIA_ROOT=/absolute/path/to/audio pnpm dev
```

配信対象は設定したディレクトリ配下の音声ファイルに限定されます。

回答と現在位置は操作のたびにブラウザの`localStorage`へ自動保存されます。画面上部の「JSONLへ保存」を押すと、回答が完了しており、前回のファイル保存後に追加または変更された項目だけが、既定で次のJSONLへ追記されます。

```text
../../poc/phoneme-alignment-evaluation/data/reviews/review-records.jsonl
```

別の保存先を使用する場合は、起動時に絶対パスまたはこのディレクトリからの相対パスで指定します。

```sh
PHONIA_REVIEW_OUTPUT=/absolute/path/to/review-records.jsonl pnpm dev
```

デフォルトではPoCで生成した次のレビューデータを読み込みます。

```text
../../poc/phoneme-alignment-evaluation/data/reviews/review-dataset.json
```

別のデータセットを既定値にする場合は、起動時に指定します。

```sh
PHONIA_REVIEW_DATASET=/absolute/path/to/review-dataset.json pnpm dev
```

この設定で配信されるのは指定したデータセット1ファイルだけです。方式名との対応表は配信されません。

一時的に別の公開URLを読み込む場合は、URLクエリで指定できます。同梱サンプルを表示する例は次のとおりです。

```text
http://localhost:5173/?dataset=/examples/review-dataset.json
```

## 確認コマンド

```sh
pnpm lint
pnpm test
pnpm typecheck
pnpm build
pnpm validate:example
pnpm validate:example -- ../../poc/phoneme-alignment-evaluation/data/reviews/review-dataset.json
```

共通レビューデータの仕様は[`docs/data-format.md`](docs/data-format.md)を参照してください。

各区間の回答が完了すると次のレビュー区間へ自動的に進みます。最終区間ではそのまま留まります。ページを再読み込みした場合は、JSONLの最新リビジョンを基準に、同じデータセットについてブラウザへ自動保存した回答と現在位置を復元します。

画面上部の「音声ファイル」から元音声を選ぶと、そのファイルに含まれる最初のレビュー区間へ移動できます。選択肢にはファイルごとの区間数も表示されます。
