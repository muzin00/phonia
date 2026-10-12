"""Attribute new closed-set rank failures using cached, audited phone weights."""

import json

import analyze as a
import numpy as np

SOURCE = a.ROOT / "artifacts/common-voice-degradation/weights-20261012-v1"
OUT = a.ROOT / "artifacts/common-voice-degradation/transitions-20261012-v1"
BEFORE, AFTER = "u16000-selected", "u64000-selected"


def descriptors(arrays):
    meta = arrays["qmeta"]
    counts = np.rint(np.expm1(meta[:, :, 0]))
    return {
        "segment_duration_seconds": (np.exp(meta[:, :, 1]) * counts).sum(axis=1),
        "segment_count": counts.sum(axis=1),
        "rms_dbfs": (meta[:, :, 3] * 50 * counts).sum(axis=1) / counts.sum(axis=1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a.pin(a.BASE / "transitions.py")
    a.pin(a.BASE / "analyze.py")
    weights_result = a.read(SOURCE / "results.json")
    original = a.read(a.RUN / "results.json")
    phones = a.read(a.RUN / "design-freeze.json")["phones"]
    output = {}
    for key in weights_result["conditions"]:
        source = original["sources"][key]
        arrays, data, info = {}, {}, {}
        for split in ("validation", "test"):
            path = a.ROOT / source["fusion"] / f"common_voice-{split}.npz"
            a.pin(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays[split] = {k: archive[k] for k in archive.files}
            data[split] = a.read(
                a.ROOT / source["fusion"] / f"common_voice-{split}-inputs.json"
            )
            info[split] = descriptors(arrays[split])
        validation_active = (
            arrays["validation"]["masks"].reshape(900, 30, -1)[:, 0, :5].all(axis=1)
        )
        cuts = {
            k: np.quantile(v[validation_active], [0.25, 0.5, 0.75])
            for k, v in info["validation"].items()
        }
        output[key] = {}
        for split in ("validation", "test"):
            path = SOURCE / f"{key}-{split}-weights.npz"
            a.pin(path)
            assert (
                a.FILES[str(path.relative_to(a.ROOT))]
                == weights_result["audit"]["files"][str(path.relative_to(a.ROOT))]
            )
            with np.load(path, allow_pickle=False) as archive:
                active, before, after = (
                    archive["active"],
                    archive[BEFORE],
                    archive[AFTER],
                )
            query_ids = np.unique(active // 30)
            assert np.array_equal(
                active, (query_ids[:, None] * 30 + np.arange(30)).ravel()
            )
            components = arrays[split]["components"][active]
            cb, ca = before * components, after * components
            scores_b, scores_a = (
                cb.sum(axis=1).reshape(-1, 30),
                ca.sum(axis=1).reshape(-1, 30),
            )
            claims = {v: i for i, v in enumerate(data[split]["speakers"])}
            own = np.array(
                [claims[data[split]["queries"][i]["speaker_id"]] for i in query_ids]
            )
            ii = np.arange(len(query_ids))
            sb, sa = scores_b.copy(), scores_a.copy()
            sb[ii, own] = sa[ii, own] = -np.inf
            hard_before, hard_after = sb.argmax(axis=1), sa.argmax(axis=1)
            mb = scores_b[ii, own] - sb.max(axis=1)
            ma = scores_a[ii, own] - sa.max(axis=1)
            new = (mb > 0) & (ma <= 0)
            recovered = (mb <= 0) & (ma > 0)
            # Fix the after-model's hardest impostor for BOTH checkpoints, so the
            # decomposition compares the same genuine and impostor pair.
            cb, ca = cb.reshape(-1, 30, len(phones)), ca.reshape(-1, 30, len(phones))
            delta_own = ca[ii, own] - cb[ii, own]
            delta_other = ca[ii, hard_after] - cb[ii, hard_after]
            delta_margin = delta_own - delta_other
            old_fixed = scores_b[ii, own] - scores_b[ii, hard_after]
            np.testing.assert_allclose(
                delta_margin.sum(axis=1), ma - old_fixed, atol=1e-12, rtol=0
            )
            assert (old_fixed[new] > 0).all()
            row = {
                "scored_queries": len(ii),
                "new_rank_failures": int(new.sum()),
                "recoveries": int(recovered.sum()),
                "new_failure_query_speakers": len(
                    {data[split]["queries"][i]["speaker_id"] for i in query_ids[new]}
                ),
                "new_failure_mean_genuine_score_change": float(
                    delta_own[new].sum(axis=1).mean()
                )
                if new.any()
                else None,
                "new_failure_mean_impostor_score_change": float(
                    delta_other[new].sum(axis=1).mean()
                )
                if new.any()
                else None,
                "new_failure_mean_margin_change_by_phone": dict(
                    zip(phones, delta_margin[new].mean(axis=0).tolist(), strict=True)
                )
                if new.any()
                else {},
                "new_failure_changed_hardest_impostor": int(
                    (new & (hard_before != hard_after)).sum()
                ),
                "strata": {},
            }
            for name, edges in cuts.items():
                bins = np.searchsorted(
                    edges, info[split][name][query_ids], side="right"
                )
                row["strata"][name] = {
                    "validation_quartile_cuts": edges.tolist(),
                    "groups": [],
                }
                for group in range(4):
                    mask = bins == group
                    row["strata"][name]["groups"].append(
                        {
                            "queries": int(mask.sum()),
                            "failures_before": int(((mb <= 0) & mask).sum()),
                            "failures_after": int(((ma <= 0) & mask).sum()),
                            "new_failures": int((new & mask).sum()),
                            "recoveries": int((recovered & mask).sum()),
                        }
                    )
            output[key][split] = row
    for path, digest in a.FILES.items():
        import hashlib

        assert hashlib.sha256((a.ROOT / path).read_bytes()).hexdigest() == digest
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "scope": "Post hoc description of selected 16k to 64k rank flips, not verification EER. Strata use validation quartiles. RMS is not recording quality. Fixed after-model hardest impostor gives an exact phone-wise margin decomposition, not a causal claim about training.",
                "conditions": output,
                "files": a.FILES,
            },
            indent=2,
        )
        + "\n"
    )
    print("Completed exact margin decomposition for", len(output), "conditions")


if __name__ == "__main__":
    main()
