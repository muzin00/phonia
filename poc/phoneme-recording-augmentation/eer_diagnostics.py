"""Paired EER uncertainty for the primary five-vowel JVS diagnostics."""

import train as e

s, np, t, qr = e.s, e.np, e.t, e.qr


def main():
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    speakers = t.jvs_data(design["source_config"], "test")["speakers"]
    lookup = {sp: i for i, sp in enumerate(speakers)}
    counts = qr.counts_for(speakers, config)
    draws = {}
    results = []
    checks = 0
    policy = next(
        p for p in design["source_config"]["policies"] if p["name"] == "vowels5"
    )
    names = [
        "original",
        *[f"{arm}-{seed}" for arm in config["arms"] for seed in config["seeds"]],
    ]
    for condition in ("clean", "white-snr20", "white-snr10", "reverb-rt60-0.3"):
        per_model = {}
        for name in names:
            rows = [
                r
                for r in s.apply_policy(
                    list(s.rows(run / "test" / name / f"{condition}-scores.jsonl")),
                    policy,
                )
                if r["status"] == "scored"
            ]
            scores = np.array([r["score"] for r in rows])
            q = np.array([lookup[r["speaker_id"]] for r in rows])
            c = np.array([lookup[r["claimed_speaker_id"]] for r in rows])
            genuine = q == c
            values = s.original.weighted_eers(scores, q, c, counts)
            for b in range(0, len(counts), max(1, len(counts) // 20)):
                w = np.where(genuine, counts[b, q], counts[b, q] * counts[b, c])
                assert (
                    abs(s.original.independent_eer(scores, genuine, w) - values[b])
                    < 1e-12
                )
                checks += 1
            per_model[name] = {
                "eer": s.original.independent_eer(scores, genuine),
                "ci95": qr.interval(values),
            }
            draws[f"{condition}/{name}"] = values
        augmented = np.mean(
            [draws[f"{condition}/augmented-{seed}"] for seed in config["seeds"]], axis=0
        )
        clean = np.mean(
            [draws[f"{condition}/clean-{seed}"] for seed in config["seeds"]], axis=0
        )
        results.append(
            {
                "condition": condition,
                "policy": "vowels5",
                "per_model": per_model,
                "augmented_minus_clean_ci95": qr.interval(augmented - clean),
                "augmented_minus_original_ci95": qr.interval(
                    augmented - draws[f"{condition}/original"]
                ),
            }
        )
        print(condition, results[-1]["augmented_minus_clean_ci95"], flush=True)
    result = {
        "scope": "post-scoring uncertainty diagnosis, no change to training or thresholds",
        "replicates": len(counts),
        "seed": config["bootstrap_seed"],
        "independent_weighted_roc_checks": checks,
        "results": results,
    }
    s.write_json(run / "eer-diagnostics.json", result)
    s.write_json(e.BASE / "eer-diagnostics.json", result)
    np.savez(run / "eer-bootstrap.npz", **draws)


if __name__ == "__main__":
    main()
