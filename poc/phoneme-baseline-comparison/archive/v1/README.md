# 比較設計1.0.0の保存記録

現行設計は[比較設計2.0.0](../../comparison-design.md)。本ディレクトリは旧設計・設定の記録である。
母音の実利用時間へECAPAを元WAVごとに合わせる2条件は、30 msの推論エラーと
推論成立性だけでは認証精度を判断できないことから撤回した。

- [旧比較設計](comparison-design.md)
- [旧protocol](config/comparison-protocol.json)
- [短入力確認時の旧設定](config/short-input-smoke.json)
- [短入力確認結果](../../short-input-results.md)

旧protocolは元のSHA-256
`15f506a6e4d5d1b357fbcfcce47208b51e3740fcb6fe8e468af6ee3450ab4a78`を保持する。
現行の短入力診断設定はこの保存済みprotocolを参照し、撤回条件を再現できる。
短入力runの当時の実装・設定は、各runの`implementation-snapshot/`にも保存した。
