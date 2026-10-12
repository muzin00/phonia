"""Label-oracle score substitution for EER attribution, never a usable model."""

import hashlib
import json

import analyze as a
import numpy as np

OUT = a.ROOT / "artifacts/common-voice-degradation/score-shift-20261012-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a.pin(a.BASE / "score_shift.py")
    a.pin(a.BASE / "analyze.py")
    a.pin(a.BASE.parent / "phoneme-speaker-encoder/phase3_train/metrics.py")
    original = a.read(a.RUN / "results.json")
    output = {}
    for key, source in original["sources"].items():
        if source["arm"] != "combined420":
            continue
        output[key] = {}
        for split in ("validation", "test"):
            output[key][split] = {}
            for mode in ("selected", "final"):
                before, after = [
                    a.read(
                        a.RUN
                        / "scores"
                        / f"common_voice-{key}-u{n}-{mode}-{split}.json"
                    )
                    for n in (16000, 64000)
                ]
                assert [a.identity(r) for r in before] == [a.identity(r) for r in after]
                assert [r["score"] is None for r in before] == [
                    r["score"] is None for r in after
                ]
                active = np.array([r["score"] is not None for r in before])
                genuine = np.array([r["is_genuine"] for r in before])[active]
                b, c = [
                    np.array([r["score"] for r in rows], dtype=float)[active]
                    for rows in (before, after)
                ]

                def eer(scores, labels=genuine):
                    return a.roc_eer(scores, labels)["eer"] * 100

                both_before, both_after = eer(b), eer(c)
                only_genuine = eer(np.where(genuine, c, b))
                only_impostor = eer(np.where(genuine, b, c))
                genuine_contribution = (
                    (only_genuine - both_before) + (both_after - only_impostor)
                ) / 2
                impostor_contribution = (
                    (only_impostor - both_before) + (both_after - only_genuine)
                ) / 2
                assert (
                    abs(
                        genuine_contribution
                        + impostor_contribution
                        - (both_after - both_before)
                    )
                    < 1e-12
                )
                output[key][split][mode] = {
                    "before_eer_percent": both_before,
                    "after_eer_percent": both_after,
                    "oracle_only_genuine_changed_eer_percent": only_genuine,
                    "oracle_only_impostor_changed_eer_percent": only_impostor,
                    "genuine_change_attribution_pp": genuine_contribution,
                    "impostor_change_attribution_pp": impostor_contribution,
                    "score_quantiles": {
                        label: {
                            period: dict(
                                zip(
                                    ("p01", "p05", "median", "p95", "p99"),
                                    np.quantile(
                                        scores[mask], [0.01, 0.05, 0.5, 0.95, 0.99]
                                    ).tolist(),
                                    strict=True,
                                )
                            )
                            for period, scores in (("before", b), ("after", c))
                        }
                        for label, mask in (
                            ("genuine", genuine),
                            ("impostor", ~genuine),
                        )
                    },
                }
    audited = a.read(a.RUN / "complete-audit.json")["files"]
    for path, digest in a.FILES.items():
        assert hashlib.sha256((a.ROOT / path).read_bytes()).hexdigest() == digest
        if path in audited:
            assert digest == audited[path]
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "scope": "Exploratory label-oracle substitution; cannot be used for inference or model selection. EER change is attributed to genuine and impostor score changes by averaging the two substitution orders (two-factor Shapley decomposition). Describes score distribution changes, not causality of training.",
                "conditions": output,
                "files": a.FILES,
            },
            indent=2,
        )
        + "\n"
    )
    print("Completed EER score-shift attribution for", len(output), "conditions")


if __name__ == "__main__":
    main()
