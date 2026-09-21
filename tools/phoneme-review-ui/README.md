# Phoneme Review UI

音声波形とモデルが推定した区間を視覚・聴覚で評価するためのローカルWeb UIです。

## 必要な環境

- Node.js 22.20.0
- npm 10以降

このディレクトリの `.node-version` にNode.jsのバージョンを記録しています。fnmを使用する場合は次のように準備します。

```sh
eval "$(fnm env --shell zsh)"
fnm use
```

## セットアップ

```sh
npm ci
npm run dev
```

Viteの開発サーバーが表示するURLをブラウザで開きます。

## 確認コマンド

```sh
npm run lint
npm run typecheck
npm run build
```
