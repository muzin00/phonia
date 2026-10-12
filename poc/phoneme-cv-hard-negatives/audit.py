"""Verify exact per-corpus schedule restriction and frozen continuation."""

import cv_only as h
import numpy as np

d, s, e, b = h.d, h.s, h.e, h.b


def load_schedule(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def main():
    config = d.read(h.CONFIG)
    run = d.ROOT / config["run_directory"]
    d.check_files(d.read(run / "design-freeze.json")["files"])
    files, counts = {}, {}
    for seed, source in h.sources(config).items():
        directory = run / f"acoustic-{seed}"
        fusion = d.ROOT / source["fusion"]
        name = f"schedule-{config['fusion_seed']}.npz"
        original = load_schedule(d.ROOT / source["budget"] / name)
        all_hard = load_schedule(
            d.ROOT / config["all_hard_run"] / f"acoustic-{seed}" / name
        )
        actual = load_schedule(directory / name)
        data = d.read(fusion / "train-inputs.json")
        cv = np.array([q["speaker_id"].startswith("cv17_") for q in data["queries"]])[
            original["queries"]
        ]
        np.testing.assert_array_equal(actual["queries"], original["queries"])
        np.testing.assert_array_equal(actual["queries"], all_hard["queries"])
        np.testing.assert_array_equal(
            actual["negative_claims"][cv], all_hard["negative_claims"][cv]
        )
        np.testing.assert_array_equal(
            actual["negative_claims"][~cv], original["negative_claims"][~cv]
        )
        for key, value in actual.items():
            np.testing.assert_array_equal(value[:4000], original[key][:4000])
        counts[str(seed)] = {
            "cv_changes": int(
                (actual["negative_claims"] != original["negative_claims"]).sum()
            ),
            "noncv_changes": int(
                (
                    actual["negative_claims"][~cv] != original["negative_claims"][~cv]
                ).sum()
            ),
            "restored_noncv": int(
                (actual["negative_claims"] != all_hard["negative_claims"]).sum()
            ),
        }
        # CV candidate identities and hardness come from the previous audited
        # experiment; this experiment only removes non-CV interventions.
        d.check_files(d.read(directory / "schedule-freeze.json")["files"])
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
                    rows = d.read(
                        run
                        / "scores"
                        / f"{corpus}-cv_only-seed{seed}-u{count}-{mode}-validation.json"
                    )
                    assert abs(e.eer(rows) - item["validation_eer"][corpus]) < 1e-12
        files.update(d.read(directory / "schedule-freeze.json")["files"])
        files.update(d.read(trained / "complete.json")["files"])
        print("Audited", seed, counts[str(seed)], flush=True)
    files.update(d.read(run / "score-audit.json")["files"])
    d.check_files(files)
    d.fixed(
        run / "complete-audit.json",
        {
            "schedule_counts": counts,
            "files": files,
            "score_audit": d.read(run / "score-audit.json"),
        },
    )


if __name__ == "__main__":
    main()
