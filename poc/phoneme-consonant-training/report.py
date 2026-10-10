#!/usr/bin/env python3
"""Save a compact, checksum-backed report after fixed-budget training."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config = read(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    summary = read(run / "training/summary.json")
    if summary["status"] != "completed" or summary["completed_updates"] != 30000:
        raise ValueError("completed fixed-budget training required")
    preparation = read(run / "preparation-report.json")
    adoption = read(run / "adoption.json")
    selections = list(summary["speaker_selection_counts"].values())
    report = {
        "status": "completed",
        "run_directory": config["run_directory"],
        "training_segments": preparation["training_segments"],
        "training_labels": preparation["training_labels"],
        "nasal_counts": preparation["nasal_counts"],
        "exclusions": preparation["exclusions"],
        "coverage": preparation["coverage"],
        "minimum_segments_per_speaker_phoneme": preparation[
            "minimum_segments_per_speaker_phoneme"
        ],
        "adoption": adoption,
        **{
            key: summary[key]
            for key in (
                "completed_updates",
                "training_examples",
                "encoder_parameters",
                "selected_update",
                "initial_encoder_sha256",
                "final_encoder_sha256",
                "phoneme_microbatch_counts",
                "validation_diagnostics",
                "export_reload_bitwise_equal_phonemes",
                "freeze_sha256",
                "test_used",
                "comparison_completed",
                "outputs_sha256",
            )
        },
        "speaker_selection_range": [min(selections), max(selections)],
    }
    (BASE / "training-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(
        f"| {r['update']:,} | {r['vowel_only_macro_eer'] * 100:.3f}% |"
        for r in summary["validation_diagnostics"]
    )
    coverage = "\n".join(
        f"| /{phone}/ | {preparation['coverage'][f'JVS/{phone}']:,} | {preparation['coverage'][f'CV/{phone}']:,} |"
        for phone in config["phonemes"]
    )
    text = f"""# 5母音＋ /m/・/n/ の学習結果

JVS＋Common Voiceの140話者ラベル・{preparation["training_segments"]:,}区間で、
7音素の共通encoderを初期値から30,000更新学習した。1条件・1 seed、65,920 parameters。
最終更新の重みを採用し、7音素それぞれでexport前後の128次元出力の完全一致を確認した。

## 入力と品質チェック

| 音素 | JVS | Common Voice |
| --- | ---: | ---: |
{coverage}

追加子音は自動チェック後72,271区間。JVS21件、Common Voice1,083件を−50dBFS未満または無音で除外した。
30ms未満による除外は0件。全140話者が全7音素を持ち、話者・音素ごとの最低区間数は{preparation["minimum_segments_per_speaker_phoneme"]}。
SRC4VCは使わず、JVS validation/test話者を学習に含めていない。

JVS/CVの事前聴取所見と今回のユーザーの学習開始指示を採用記録へ残した。
レビュー済み候補サンプルのうち自動チェック後に残ったのはJVS50/50、CV97/100件。
この件数は人手の回答済み件数・合格率ではない。個別回答JSONLと定量的な人手品質率はなく、
通常手順の自動チェック後100件／コーパスの追加レビューは実施していない。
採用条件と手順差分は[README](README.md)と結果JSONの`adoption`に記録した。

## 学習とvalidation診断

1更新100区間、総投入3,000,000区間。7音素のうち5音素を順に選択し、各音素の選択回数の差は最大1回。
各話者の選択回数は{min(selections):,}〜{max(selections):,}回。
早期終了・validationによる重みの選び直しは行わず、30,000更新の最終重みを固定した。

| 更新数 | 母音単一区間validation macro EER |
| ---: | ---: |
{rows}

この指標は既存5母音の固定JVS validationによる診断で、m/nの性能や発話単位のFAR・FRR・EERを表さない。
testは未使用。子音追加による改善の有無は、同じencoderと使用PCM秒数を揃えた後続の比較で確認する。

## 成果物と検証

生成先: `{config["run_directory"]}`

- `bundle/encoder.pt`: 7音素共通encoder。
- `bundle/feature-statistics.json`: trainのみで計算した特徴統計。
- `training/last.pt`: optimizer・RNG・サンプラーを含む再開用checkpoint。
- `training/history.jsonl`: 30,000更新の学習履歴。
- `training-freeze.json`: 入力・コード・設定・元WAVのSHA-256。
- [training-results.json](training-results.json): 条件・集計・モデルと成果物のSHA-256。

音素カバレッジ・正例ペア・欠損拒否・checkpoint再開の一致を検証する4テストを実施した。
学習前後の凍結checksumを照合し、全7音素のexport再読込結果が完全一致した。
"""
    (BASE / "training-results.md").write_text(text, encoding="utf-8")
    print(f"saved {BASE / 'training-results.md'}")


if __name__ == "__main__":
    main()
