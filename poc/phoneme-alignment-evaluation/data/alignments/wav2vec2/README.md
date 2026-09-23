# Wav2Vec2 phoneme CTC ONNX results

固定した`facebook/wav2vec2-xlsr-53-espeak-cv-ft`をFP32 ONNXへ変換し、
共通期待音素列に対してCTC forced alignmentを実行した結果を保存する。

- `export-validation.json`: PyTorchとONNXの同等性検証
- `raw/`: 期待音素全体のCTC境界、スコア、制約なし推論結果
- `normalized.jsonl`: 共通形式へ正規化した原子母音区間
- `validation.json`: 音素列と区間の構造検査
- `comparison.json`: Juliusの同一原子母音区間との機械比較
- `run.json`: モデル、実行環境、処理時間、制約なし推論の誤り率

構造検査では138母音区間を生成できたが、制約なし推論の期待トークン列に対する
編集誤り率は40.4%だった。Juliusとの138区間比較では重なりなしが12件あり、
そのうち11件はJuliusの人間レビューで利用可能とされた区間だった。

一方、少数区間を実際に聴いた参考確認では明らかな品質不足は感じられず、聴感上は
MFAと同程度という印象だった。方式間の境界不一致と制約なし認識の誤り率だけでは、
forced alignment後の母音区間の利用可否を確定できない。

Wav2Vec2の全138件を対象とする定量的な人間レビューは実施しない。今回の結果は
「機械指標ではJuliusより不利、少数の聴感確認ではMFAと同程度」と扱う。Juliusを
上回る品質上の根拠がなく、処理時間も約13.3倍であるため、基準方式にはJuliusを
採用する。詳細な数値は`comparison.json`を参照する。
