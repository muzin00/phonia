"""Read-only diagnosis of frozen CV scores; no test-based model selection."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-speaker-encoder/phase3_train"))
from metrics import roc_eer

RUN = ROOT / "artifacts/phoneme-data-budget/replicated280420-20261012-v1"
OUT = ROOT / "artifacts/common-voice-degradation/frozen-scores-20261012-v1"
FILES = {}


def pin(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = str(path.relative_to(ROOT))
    if key in FILES:
        assert FILES[key] == digest
    FILES[key] = digest


def read(path):
    pin(path)
    return json.loads(path.read_text())


def eer(rows):
    active = [r for r in rows if r["score"] is not None]
    return (
        100
        * roc_eer([r["score"] for r in active], [r["is_genuine"] for r in active])[
            "eer"
        ]
    )


def identity(r):
    return tuple(
        r[k]
        for k in ("query_id", "speaker_id", "claimed_speaker_id", "role", "is_genuine")
    )


def query_metrics(rows):
    groups = {}
    for row in rows:
        if row["score"] is not None:
            groups.setdefault(row["query_id"], []).append(row)
    result = {}
    for query, values in groups.items():
        own = [r for r in values if r["is_genuine"]]
        assert len(own) == 1
        impostors = [r["score"] for r in values if not r["is_genuine"]]
        margin = own[0]["score"] - max(impostors)
        result[query] = {
            "speaker": own[0]["speaker_id"],
            "margin": margin,
            "rank_failure": int(margin <= 0),
        }
    return result


def paired(before, after):
    assert [identity(r) for r in before] == [identity(r) for r in after]
    assert [r["score"] is None for r in before] == [r["score"] is None for r in after]
    speakers = sorted({r["speaker_id"] for r in before})
    a, b = query_metrics(before), query_metrics(after)
    rows = []
    for speaker in speakers:
        ids = [q for q in a if a[q]["speaker"] == speaker]
        removed_before = [r for r in before if r["speaker_id"] != speaker]
        removed_after = [r for r in after if r["speaker_id"] != speaker]
        rows.append(
            {
                "speaker_id": speaker,
                "scored_queries": len(ids),
                "rank_failures_before": sum(a[q]["rank_failure"] for q in ids),
                "rank_failures_after": sum(b[q]["rank_failure"] for q in ids),
                "leave_query_speaker_out_delta_pp": eer(removed_after)
                - eer(removed_before),
            }
        )
    return {
        "eer_delta_pp": eer(after) - eer(before),
        "new_rank_failures": sum(
            not a[q]["rank_failure"] and b[q]["rank_failure"] for q in a
        ),
        "recovered_rank_failures": sum(
            a[q]["rank_failure"] and not b[q]["rank_failure"] for q in a
        ),
        "rank_failures_before": sum(v["rank_failure"] for v in a.values()),
        "rank_failures_after": sum(v["rank_failure"] for v in b.values()),
        "speakers": rows,
    }


def describe(values):
    return dict(
        zip(
            ("p10", "median", "p90"),
            np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
            strict=True,
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pin(Path(__file__))
    pin(BASE.parent / "phoneme-speaker-encoder/phase3_train/metrics.py")
    audited = read(RUN / "complete-audit.json")["files"]
    results = read(RUN / "results.json")
    conditions, contrasts, split_info = {}, {}, {}
    reference_rows = {}
    maximum, record_count = 0.0, 0
    for key, source in results["sources"].items():
        history = read(ROOT / source["budget"] / "seed-20261015/history.json")
        assert [v["update"] for v in history] == list(range(0, 64001, 500))
        conditions[key] = {"source": source, "checkpoints": {}}
        cache = {}
        for variant in (
            "equal",
            "u16000-selected",
            "u16000-final",
            "u64000-selected",
            "u64000-final",
        ):
            entry = {}
            if variant != "equal":
                count, mode = variant.split("-")
                eligible = [h for h in history if h["update"] <= int(count[1:])]
                chosen = (
                    min(eligible, key=lambda h: (h["macro_eer"], h["update"]))
                    if mode == "selected"
                    else eligible[-1]
                )
                cvbest = min(
                    eligible,
                    key=lambda h: (h["validation_eer"]["common_voice"], h["update"]),
                )
                entry.update(
                    {
                        "update": chosen["update"],
                        "validation_corpora_eer_percent": {
                            c: v * 100 for c, v in chosen["validation_eer"].items()
                        },
                        "macro_validation_eer_percent": chosen["macro_eer"] * 100,
                        "cv_best_validation_update": cvbest["update"],
                        "cv_selection_penalty_pp": 100
                        * (
                            chosen["validation_eer"]["common_voice"]
                            - cvbest["validation_eer"]["common_voice"]
                        ),
                        "candidate_count": len(eligible),
                    }
                )
            for split in ("validation", "test"):
                rows = read(
                    RUN / "scores" / f"common_voice-{key}-{variant}-{split}.json"
                )
                assert all(r["role"] == "verification" for r in rows)
                row_signature = [(identity(r), r["score"] is None) for r in rows]
                if split not in reference_rows:
                    reference_rows[split] = row_signature
                assert row_signature == reference_rows[split]
                record_count += len(rows)
                value = eer(rows)
                entry[f"{split}_eer_percent"] = value
                if split == "test":
                    expected = (
                        results["conditions"]["common_voice"][f"{key}-{variant}"][
                            "verification"
                        ]["eer"]
                        * 100
                    )
                    cache[variant] = rows
                elif variant != "equal":
                    expected = chosen["validation_eer"]["common_voice"] * 100
                else:
                    expected = history[0]["validation_eer"]["common_voice"] * 100
                maximum = max(maximum, abs(value - expected))
                assert abs(value - expected) < 1e-10
            conditions[key]["checkpoints"][variant] = entry
        if source["arm"] == "combined420":
            contrasts[key] = {
                "selected_16k_to_64k": paired(
                    cache["u16000-selected"], cache["u64000-selected"]
                ),
                "final_16k_to_64k": paired(
                    cache["u16000-final"], cache["u64000-final"]
                ),
                "selected_to_final_64k": paired(
                    cache["u64000-selected"], cache["u64000-final"]
                ),
            }
        print(key, "verified", flush=True)
    first = next(iter(results["sources"].values()))
    speaker_sets = {}
    for split in ("validation", "test"):
        data = read(ROOT / first["fusion"] / f"common_voice-{split}-inputs.json")
        path = ROOT / first["fusion"] / f"common_voice-{split}.npz"
        pin(path)
        with np.load(path, allow_pickle=False) as arrays:
            meta, masks = arrays["qmeta"], arrays["qmask"]
            # These describe extracted segments, not whole-waveform SNR or noise.
            counts = np.rint(np.expm1(meta[:, :, 0]))
            totals = counts.sum(axis=1)
            active = (
                arrays["masks"]
                .reshape(len(data["queries"]), len(data["speakers"]), -1)[:, 0, :5]
                .all(axis=1)
            )
            values = {
                "phone_types": masks.sum(axis=1),
                "segments": totals,
                "segment_duration_sum_seconds": (np.exp(meta[:, :, 1]) * counts).sum(
                    axis=1
                ),
                "segment_mean_rms_dbfs": (meta[:, :, 3] * 50 * counts).sum(axis=1)
                / totals,
            }
            split_info[split] = {
                "speakers": len(data["speakers"]),
                "queries": len(data["queries"]),
                "scored_queries": int(active.sum()),
                "scored_query_descriptors": {
                    k: describe(v[active]) for k, v in values.items()
                },
            }
            reference = reference_rows[split]
            assert len(data["queries"]) * len(data["speakers"]) == len(reference)
            assert int(active.sum()) * len(data["speakers"]) == sum(
                not missing for _, missing in reference
            )
        speaker_sets[split] = set(data["speakers"])
    assert not speaker_sets["validation"] & speaker_sets["test"]
    for path, digest in FILES.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        if path in audited:
            assert digest == audited[path]
    summary = {
        "scope": "Previously observed tests; descriptive diagnosis only. No retraining, new thresholds, or model promotion. Leave-query-speaker-out keeps the enrollment gallery fixed and is sensitivity analysis, not a confidence interval. Rank failure is auxiliary closed-set identification, not verification EER.",
        "conditions": conditions,
        "paired_test_diagnostics": contrasts,
        "split_descriptors": split_info,
        "audit": {
            "score_records": record_count,
            "maximum_eer_difference_pp": maximum,
            "files": FILES,
            "previously_audited_files_matched": sum(p in audited for p in FILES),
        },
    }
    (OUT / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Frozen-score Common Voice diagnosis",
        "",
        summary["scope"],
        "",
        "| arm | encoder | seed | model | update | validation EER % | test EER % | CV selection penalty pp |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for value in conditions.values():
        src = value["source"]
        for variant, item in value["checkpoints"].items():
            lines.append(
                f"| {src['arm']} | {src['encoder_updates']} | {src['encoder_seed']} | {variant} | {item.get('update', 0)} | {item['validation_eer_percent']:.4f} | {item['test_eer_percent']:.4f} | {item.get('cv_selection_penalty_pp', 0):.4f} |"
            )
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["audit"] | {"files": len(FILES)}, indent=2))


if __name__ == "__main__":
    main()
