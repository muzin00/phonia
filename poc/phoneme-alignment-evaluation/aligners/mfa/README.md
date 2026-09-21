# Montreal Forced Aligner

Apple Silicon上でMontreal Forced Alignerをネイティブ実行するための環境定義を管理する。
Pythonの共通処理に使用するuv環境とは分離し、Kaldiを含むネイティブ依存関係はCondaで管理する。

## 環境構築

[Miniforge](https://github.com/conda-forge/miniforge)のApple Silicon向けインストーラーを導入した後、このディレクトリで次を実行する。

Miniforgeをバッチインストールして`conda`が現在のシェルに登録されていない場合は、先に初期化スクリプトを読み込む。

```shell
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
```

```shell
conda env create --file environment.yml
conda activate phonia-mfa
```

環境が`osx-arm64`として作成され、固定したバージョンが使われていることを確認する。

```shell
conda info
mfa version
conda list --explicit
```

## 日本語モデル

音響モデルと、その学習に使用された発音辞書を取得する。

日本語テキストの分かち書きには、環境定義に含めたspaCy、SudachiPy、Sudachi辞書を使用する。これらがない場合、MFAは日本語コーパスの読み込み時に停止する。

```shell
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
```

取得後、モデル一覧を確認する。

```shell
mfa model list acoustic
mfa model list dictionary
```

モデルはMFAのデフォルトでは`~/Documents/MFA/pretrained_models/`以下に保存される。実験結果にはモデル名だけでなく、MFAが表示するモデルバージョンも記録する。
