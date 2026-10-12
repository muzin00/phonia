"""Audit fixed schedules, shared prefixes, selection and new-seed inference."""

import replication as x

d, s, e, b = x.d, x.s, x.e, x.b


def main():
    config = d.read(x.CONFIG)
    run = d.ROOT / config["run_directory"]
    d.check_files(d.read(run / "design-freeze.json")["files"])
    files, selections = {}, []
    for acoustic, source in x.sources(config).items():
        original_fusion = d.ROOT / source["fusion"]
        trained_prefixes = []
        for fusion in config["new_fusion_seeds"]:
            prefix = x.prefix_path(run, acoustic, fusion)
            prefix_files = d.read(prefix / "complete.json")["files"]
            d.check_files(prefix_files)
            files.update(prefix_files)
            d.check_files(d.read(prefix / "prepared-freeze.json")["files"])
            options = d.read(prefix / "design-freeze.json")["config"]
            original = d.read(original_fusion / "design-freeze.json")["config"]
            assert {
                k: v for k, v in options.items() if k not in ("seeds", "architectures")
            } == {
                k: v for k, v in original.items() if k not in ("seeds", "architectures")
            }
            assert options["seeds"] == [fusion]
            first = s.torch.load(
                prefix / f"transformer-{fusion}/last.pt", weights_only=False
            )
            assert first["update"] == config["budgets"][0]
            assert all(
                int(v["step"]) == config["budgets"][0]
                for v in first["optimizer"]["state"].values()
            )
            trained_prefixes.append(first["model"])
            for arm in config["arms"]:
                path = x.arm_path(run, acoustic, fusion, arm)
                actual_schedule = path / f"schedule-{fusion}.npz"
                assert d.sha(actual_schedule) == d.sha(
                    x.schedule_path(config, acoustic, arm)
                )
                with (
                    s.np.load(actual_schedule, allow_pickle=False) as actual,
                    s.np.load(
                        prefix / f"schedule-{fusion}.npz", allow_pickle=False
                    ) as short,
                ):
                    for key in actual.files:
                        s.np.testing.assert_array_equal(
                            actual[key][: config["budgets"][0]], short[key]
                        )
                trained = path / f"seed-{fusion}"
                complete = d.read(trained / "complete.json")
                d.check_files(complete["files"])
                files.update(complete["files"])
                files.update(d.pin([path / "design-freeze.json", actual_schedule]))
                last = s.torch.load(trained / "last.pt", weights_only=False)
                assert last["update"] == 64000
                assert all(
                    int(v["step"]) == 64000 for v in last["optimizer"]["state"].values()
                )
                assert all(
                    v["lr"] == config["continuation_learning_rate"]
                    for v in last["optimizer"]["param_groups"]
                )
                history = d.read(trained / "history.json")
                assert history[: len(first["history"])] == first["history"]
                assert [v["update"] for v in history] == list(range(0, 64001, 500))
                prefix_final = s.torch.load(
                    trained / "budget-4000/final.pt", weights_only=True
                )
                assert all(
                    s.torch.equal(v, prefix_final["state_dict"][k])
                    for k, v in first["model"].items()
                )
                for budget in config["budgets"][1:]:
                    eligible = [v for v in history if v["update"] <= budget]
                    best = b.select_best(eligible)
                    summary = d.read(trained / f"budget-{budget}/summary.json")
                    assert summary["selected_update"] == best["update"]
                    selections.append(
                        {
                            "acoustic": acoustic,
                            "fusion": fusion,
                            "arm": arm,
                            "budget": budget,
                            "selected": best["update"],
                        }
                    )
                    for mode, item in (("selected", best), ("final", eligible[-1])):
                        bundle = s.torch.load(
                            trained / f"budget-{budget}/{mode}.pt", weights_only=True
                        )
                        assert (
                            bundle["update"] == item["update"]
                            and bundle["seed"] == fusion
                        )
                        if mode == "final" and budget == 64000:
                            assert all(
                                s.torch.equal(v, last["model"][k])
                                for k, v in bundle["state_dict"].items()
                            )
                        label = f"{arm}-seed{acoustic}-fusion{fusion}-u{budget}-{mode}"
                        for corpus in s.CORPORA:
                            rows = d.read(
                                run / "scores" / f"{corpus}-{label}-validation.json"
                            )
                            assert (
                                abs(e.eer(rows) - item["validation_eer"][corpus])
                                < 1e-12
                            )
            print("Audited", acoustic, fusion, flush=True)
        assert any(
            not s.torch.equal(v, trained_prefixes[1][k])
            for k, v in trained_prefixes[0].items()
        )
    score = d.read(run / "score-audit.json")
    assert score["independently_inferred_records"] == 5472000
    assert score["previously_audited_records"] == 2736000
    assert score["max_difference"] == 0
    files.update(score["files"])
    files.update(d.pin([run / "all-models-freeze.json", run / "score-audit.json"]))
    d.check_files(files)
    d.fixed(
        run / "complete-audit.json",
        {"files": files, "selections": selections, "score_audit": score},
    )


if __name__ == "__main__":
    main()
