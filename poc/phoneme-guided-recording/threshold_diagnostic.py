"""Post-scoring diagnostic: reuse single-clip thresholds without tuning them."""

import experiment as e


def main():
    s, qr = e.s, e.qr
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    frozen = s.read_json(run / "threshold-freeze.json")
    results = []
    for corpus in config["cv_runs"]:
        data, *_ = e.load(config, design, corpus, "test")
        counts = qr.counts_for(data["speakers"], config)
        baseline = list(s.rows(run / "test" / corpus / "single-scores.jsonl"))
        adaptive = list(s.rows(run / "test" / corpus / "adaptive1-scores.jsonl"))
        for op in config["operating_points"]:
            threshold = frozen["thresholds"][f"{corpus}/single"][op]
            before, b = qr.metrics(baseline, threshold, data["speakers"], counts)
            after, a = qr.metrics(adaptive, threshold, data["speakers"], counts)
            results.append(
                {
                    "corpus": corpus,
                    "operating_point": op,
                    "single": before,
                    "adaptive1_same_threshold": after,
                    "difference_ci95": {
                        key: qr.interval(a[key] - b[key]) for key in ("far", "frr")
                    },
                }
            )
    value = {
        "scope": "Post-scoring attribution diagnostic; existing single-arm thresholds transferred unchanged; no tuning, candidate selection, or replacement of primary gates",
        "results": results,
    }
    s.write_json(run / "threshold-diagnostic.json", value)
    s.write_json(e.BASE / "threshold-diagnostic.json", value)
    print(
        [
            (r["corpus"], r["operating_point"], r["adaptive1_same_threshold"]["frr"])
            for r in results
        ],
        flush=True,
    )


if __name__ == "__main__":
    main()
