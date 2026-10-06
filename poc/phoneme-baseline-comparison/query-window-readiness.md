# 改訂設計2.0.0のvalidation入力確認

2026-10-06に、照合音声の上限長full・1・2・3・5秒をvalidation metadataで確認した。
登録入力は固定したまま、同じ中心の波形区間を全3方式へ渡す。
音声・モデル・認証score・testの派生入力は読み込んでいない。
本書は入力条件の確認結果であり、認証性能の結果ではない。

## 母音方式の入力充足

元母音の境界全体が共通波形内にある区間だけを採用した。
境界内方式と文脈付き方式は同じanchor集合を使うため、5母音の充足条件も同じである。

| 照合波形の上限 | 通常文の5母音充足数 / 全query | 充足率 | 異文の5母音充足数 / 全query | 充足率 |
| --- | --- | --- | --- | --- |
| 1秒 | 47 / 750 | 6.27% | 28 / 450 | 6.22% |
| 2秒 | 245 / 750 | 32.67% | 192 / 450 | 42.67% |
| 3秒 | 501 / 750 | 66.80% | 324 / 450 | 72.00% |
| 5秒 | 681 / 750 | 90.80% | 382 / 450 | 84.89% |
| full | 735 / 750 | 98.00% | 392 / 450 | 87.11% |

全条件で同じ1,200 queryを保持し、母音不足を全入力FRRの分母から除かない。
この充足率はmetadata上での値。ECAPAのscore取得率と全方式の認証精度は未測定である。
1秒の異文条件には充足queryが0件の話者がある。
全入力FRR・coverageは保存し、その条件の母音方式の条件付き指標・曲線・信頼区間は評価不能として扱う。

## 指定上限より短い元発話

| 条件 | 通常文で上限より短い元発話 | 異文で上限より短い元発話 | 実際の照合波形長 |
| --- | --- | --- | --- |
| 1秒 | 0 | 0 | 全queryが1秒 |
| 2秒 | 0 | 0 | 全queryが2秒 |
| 3秒 | 0 | 32 | 通常文は3秒、異文は2.272〜3秒 |
| 5秒 | 38 | 273 | 通常文は3.539〜5秒、異文は2.272〜5秒 |

元発話を延長・反復・他発話と連結しない。
長さ曲線の横軸は「最大照合波形長」であり、実際に渡す長さの分布も併記する。
元発話が5秒以上ある固定集合は通常文712件、異文177件。
この集合でも全上限条件を比較し、母音不足のqueryを保持する。
閾値は各上限の全query較正値を再利用し、固定集合だけで較正し直さない。

## 区間と境界の検証

各query・上限について、元WAV・checksum、共通波形のframe境界、実時間、
採用anchorと除外anchor、境界内・文脈付きの実効frame境界、和集合・重複込みの利用量を保存した。
共通波形は入れ子。文脈はその波形の外側を使用しない。
短縮後のalignmentは再推定せず、元発話の固定alignment metadataを使用する。
このため短い録音から音素抽出するend-to-end性能は本集計の対象に含めない。

## 再現と成果物

```sh
python3 poc/phoneme-baseline-comparison/scripts/inspect_query_windows.py \
  --output-dir artifacts/phoneme-baseline-comparison/design/<unused-run-id>
```

標準ライブラリだけで、[現行protocol](config/comparison-protocol.json)の4入力のSHA-256と
既存の話者・role・登録・照合の整合性を検証し、未使用ディレクトリへ保存する。
出力は上限ごとの`query-windows-<condition>.jsonl`各1,200行と、集計・設定・実装のhashを持つ
`query-window-readiness.json`。

記録元は`artifacts/phoneme-baseline-comparison/design/validation-query-windows-v2-r3/`。
認証用のprofile・score・閾値作成は、[比較設計](comparison-design.md)のvalidation pilot以降で行う。

記録のSHA-256は`e439252b702353aa51b04f79ae0a66c6c6eb2b6ca91176c596dec7099c770ba9`、
参照protocolは`5169ea422bf53ca11d52b47591befabaed8def7ee2d238de3ef9a0b61167c4ba`。
