# JVSデータ調査

## 1. 目的

Phase 2で利用するJVS corpusの実データを機械的に走査し、設計上の仕様と配布物の差、
学習データとして利用できる発話数、音声形式、書き起こしとの対応、既知の制約を確認する。

本調査では個別音声の品質による選別や修正を行わない。原本の状態と、機械的に
学習入力を構成できない箇所を記録する。

## 2. 調査対象

- 配布物: `jvs_ver1.zip`
- 取得元: [JVS corpus公式ページ](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- 取得日: 2026-09-26
- ZIP SHA-256: `37180e2f87bd1a3e668d7c020378f77cebf61dd57d4d74c71eb0114f386a3999`
- ZIPサイズ: 約3.54 GB
- 展開後サイズ: 約5.1 GB

公式ページと配布物の`README.txt`では、100話者について`parallel100`、`nonpara30`、
`whisper10`、`falset10`の4サブセットを提供すると説明されている。音声は24 kHz・
16-bitのスタジオ録音である。

## 3. 利用条件

配布物の`README.txt`と公式ページには、次の条件が記載されている。

- 音声は学術研究、非商用研究、個人利用で利用できる。
- 音声データの再配布は許可されていない。
- タグはCC BY-SA 4.0で提供される。
- テキストのライセンスは由来となるJSUT corpusの条件に従う。
- 商用利用には提供元への問い合わせが必要である。

このため、音声と配布ZIPはGit管理しない。初期PoCの研究利用と、将来の商用プロダクトで
必要となる利用許諾を区別する。

## 4. 調査方法

原本は次へ配置する。`data/source/`は`.gitignore`で除外する。

```text
data/source/jvs_ver1.zip
data/source/jvs_ver1/
```

取得と調査は次の手順で再実行できる。

```bash
cd poc/phoneme-speaker-dataset
mkdir -p data/source
uvx --from gdown gdown 19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt \
  -O data/source/jvs_ver1.zip
shasum -a 256 data/source/jvs_ver1.zip
unzip -q data/source/jvs_ver1.zip -d data/source
python3 -m scripts.inspect_jvs_corpus
```

調査スクリプトは次を話者・サブセット単位で検査する。

- 期待する100話者と4サブセットの存在
- WAV数、書き起こし数、WAVと書き起こしIDの対応
- WAVの読み出し可否、サンプルレート、チャンネル数、量子化ビット数
- 音声時間
- 配布物に含まれるモノフォン・フルコンテキストラベル数

完全な機械可読結果は[`data/jvs-inventory.json`](data/jvs-inventory.json)に保存する。

## 5. 初期調査結果

### 5.1 全体

- 話者ディレクトリ: 100/100
- 提供属性上の構成: M 49話者、F 51話者
- WAV: 14,997件
- 合計時間: 約30.25時間
- WAV形式: 全14,997件が24 kHz・mono・16-bit PCM
- 読み出し不能なWAV: 0件

話者属性は配布物の`gender_f0range.txt`を集計した値であり、音声から推定したものではない。

### 5.2 サブセット別

| サブセット | WAV | 時間 | WAVと書き起こしIDが対応 | monラベル | fulラベル |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parallel100` | 9,997 | 約21.86時間 | 9,997 | 9,795 | 9,795 |
| `nonpara30` | 3,000 | 約4.36時間 | 3,000 | 2,939 | 2,939 |
| `whisper10` | 1,000 | 約2.06時間 | 1,000 | 0 | 0 |
| `falset10` | 1,000 | 約1.96時間 | 985 | 0 | 0 |

通常発話の`parallel100`と`nonpara30`は合計12,997件、約26.22時間である。

`nonpara30`の`transcripts_utf8.txt`には各話者50件の候補文が記録され、そのうち30件に
対応するWAVがある。この構成を欠損とは扱わず、WAVと対応する30件を利用可能な発話として
数える。

### 5.3 配布物と設計上の差

`parallel100`は設計上10,000件だが、次の3件のWAVがなく、書き起こしだけが存在する。

- `jvs030/VOICEACTRESS100_045`
- `jvs074/VOICEACTRESS100_094`
- `jvs089/VOICEACTRESS100_019`

`falset10`には音声内容の欠損ではなく、WAVファイル名と書き起こしIDの不一致がある。

- `jvs001`: 5件ずつ不一致
- `jvs088`: 10件ずつ不一致

初期PoCの主対象として検討している通常発話には、`parallel100`の3件以外で
WAVと書き起こしの不一致はない。

配布済み音素ラベルは通常発話の全WAVには存在しない。Phase 2では配布ラベルを
学習区間として直接使用せず、Phase 1で採用した共通手順によりJuliusで全対象発話を
再処理するため、配布ラベルの欠損は学習区間の除外理由にしない。

## 6. 制約と次の判断

配布物からは話者ID、発話、提供属性、音声時間を取得できる一方、収録日、個別の
収録セッションID、マイクIDは確認できない。このため、JVSだけを使った初期PoCでは、
登録用と照合用を別発話へ分けることはできても、別日・別端末のクロスセッション性能を
評価できるとはみなさない。

この結果を基に、次を別文書で決定する。

- 初期学習に使用するサブセットと発話スタイル
- train、validation、testの話者分割
- 同じ文章を話者間で共有する`parallel100`の扱い
- `nonpara30`を学習・登録・照合へ割り当てる方法
- JVSの範囲で評価できる主張と、追加データが必要な評価条件
