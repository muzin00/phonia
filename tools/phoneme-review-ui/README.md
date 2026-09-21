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

デフォルトでは同梱したサンプルデータを読み込みます。別のデータセットはURLクエリで指定できます。

```text
http://localhost:5173/?dataset=/path/to/review-dataset.json
```

## 確認コマンド

```sh
pnpm lint
pnpm test
pnpm typecheck
pnpm build
pnpm validate:example
```

共通レビューデータの仕様は[`docs/data-format.md`](docs/data-format.md)を参照してください。
