"""Audit train-only mining, preserved schedules, optimizer and model selection."""

import hard_negative as h
import numpy as np

d, s, e, b = h.d, h.s, h.e, h.b


def main():
    config = d.read(h.CONFIG)
    run = d.ROOT / config["run_directory"]
    d.check_files(d.read(run / "design-freeze.json")["files"])
    files = {}
    mining_rows = 0
    for seed, source in h.sources(config).items():
        directory = run / f"acoustic-{seed}"
        fusion = d.ROOT / source["fusion"]
        data = d.read(fusion / "train-inputs.json")
        d.check_files(d.read(directory / "schedule-freeze.json")["files"])
        with np.load(fusion / "train.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        with np.load(
            d.ROOT / source["budget"] / f"schedule-{config['fusion_seed']}.npz",
            allow_pickle=False,
        ) as archive:
            original = {k: archive[k].copy() for k in archive.files}
        with np.load(
            directory / f"schedule-{config['fusion_seed']}.npz", allow_pickle=False
        ) as archive:
            actual = {k: archive[k].copy() for k in archive.files}
        table = np.load(directory / "mining-scores.npy", allow_pickle=False)
        owner = np.array(
            [data["speakers"].index(v["speaker_id"]) for v in data["queries"]]
        )
        # Direct per-query sum is independent of the matrix multiplication miner.
        maxdiff = 0.0
        for q in range(len(owner)):
            candidates = np.flatnonzero(np.isfinite(table[q]))
            mask = (
                arrays["emask"][candidates]
                & arrays["qmask"][q]
                & arrays["emask"][owner[q]]
            )
            values = (
                arrays["enrollment"][candidates].astype(np.float64)
                * arrays["queries"][q].astype(np.float64)
            ).sum(axis=2)
            scores = (values * mask).sum(axis=1) / mask.sum(axis=1)
            maxdiff = max(maxdiff, float(np.max(np.abs(scores - table[q, candidates]))))
            assert owner[q] not in candidates
            assert all(
                h.mining.corpus(data["speakers"][c])
                == h.mining.corpus(data["speakers"][owner[q]])
                for c in candidates
            )
        assert maxdiff < 1e-12
        expected, diagnostic = h.mining.make_schedule(original, table, owner, config)
        with np.load(
            directory / "mining-candidates.npz", allow_pickle=False
        ) as archive:
            for key in diagnostic:
                np.testing.assert_array_equal(archive[key], diagnostic[key])
        for key, value in actual.items():
            np.testing.assert_array_equal(value, expected[key])
            np.testing.assert_array_equal(value[:4000], original[key][:4000])
        np.testing.assert_array_equal(actual["queries"], original["queries"])
        mining_rows += len(diagnostic["positions"])
        trained = directory / f"seed-{config['fusion_seed']}"
        d.check_files(d.read(trained / "complete.json")["files"])
        last = s.torch.load(trained / "last.pt", weights_only=False)
        assert last["update"] == 64000
        assert all(int(v["step"]) == 64000 for v in last["optimizer"]["state"].values())
        assert all(
            v["lr"] == config["continuation_learning_rate"]
            for v in last["optimizer"]["param_groups"]
        )
        history = d.read(trained / "history.json")
        prefix = d.read(fusion / f"transformer-{config['fusion_seed']}/history.json")
        assert history[: len(prefix)] == prefix
        assert [v["update"] for v in history] == list(range(0, 64001, 500))
        for count in (16000, 64000):
            eligible = [v for v in history if v["update"] <= count]
            best = b.select_best(eligible)
            summary = d.read(trained / f"budget-{count}/summary.json")
            assert summary["selected_update"] == best["update"]
            for mode, item in (("selected", best), ("final", eligible[-1])):
                bundle = s.torch.load(
                    trained / f"budget-{count}/{mode}.pt", weights_only=True
                )
                assert bundle["update"] == item["update"]
                if mode == "final" and count == 64000:
                    assert all(
                        s.torch.equal(v, last["model"][k])
                        for k, v in bundle["state_dict"].items()
                    )
                for corpus in s.CORPORA:
                    label = f"mixed-seed{seed}-u{count}-{mode}"
                    rows = d.read(run / "scores" / f"{corpus}-{label}-validation.json")
                    assert abs(e.eer(rows) - item["validation_eer"][corpus]) < 1e-12
        files.update(d.read(directory / "schedule-freeze.json")["files"])
        files.update(d.read(trained / "complete.json")["files"])
        print("Audited", seed, "mining max difference", maxdiff, flush=True)
    files.update(d.read(run / "score-audit.json")["files"])
    d.check_files(files)
    d.fixed(
        run / "complete-audit.json",
        {
            "hard_slots_verified": mining_rows,
            "files": files,
            "score_audit": d.read(run / "score-audit.json"),
        },
    )


if __name__ == "__main__":
    main()
