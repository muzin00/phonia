"""Recompute saved candidate EERs, search decisions and evaluation metrics."""

from __future__ import annotations

from itertools import pairwise

from core import (
    BASE,
    ROOT,
    check,
    independent_eer,
    load_inputs,
    metrics,
    np,
    pin,
    positions,
    read,
    search,
    sha,
    subset_id,
    write,
)


def main():
    config = read(BASE / "config.json")
    run = ROOT / config["run_directory"]
    frozen = read(run / "design-freeze.json")
    winner = read(run / "winner-freeze.json")
    selection = read(run / "selection-freeze.json")
    assert read(run / "state.json")["status"] == "complete"
    check(frozen["files"])
    check(read(run / "complete-audit.json")["files"])
    assert winner["design_sha256"] == sha(run / "design-freeze.json")
    assert selection["winner_sha256"] == sha(run / "winner-freeze.json")
    recorded = read(run / "search-results.json")
    candidates = {tuple(row["subset"]): row for row in recorded["candidates"]}
    assert len(candidates) == len(recorded["candidates"]) == winner["candidates"]
    inspected = set()

    def evaluate(subsets):
        inspected.update(subsets)
        return {v: candidates[v]["objective"] for v in subsets}

    actual, history = search(frozen["phones"], evaluate, config)
    assert list(actual) == winner["subset"]
    assert history == recorded["history"]
    assert inspected == set(candidates)
    assert all(a["objective"] > b["objective"] for a, b in pairwise(history))
    total_discovery_scores, total_evaluation_scores = 0, 0
    recomputed = {v: [] for v in candidates}
    max_error = 0.0
    for seed in config["encoder_seeds"]:
        discovery_ids, discovery_labels = {}, {}
        for corpus in config["corpora"]:
            arrays, data, rows = load_inputs(config, seed, corpus, "validation")
            groups = frozen["partitions"][corpus]
            assert not set(groups["discovery"]) & set(groups["confirmation"])
            assert set(groups["discovery"]) | set(groups["confirmation"]) == set(
                data["speakers"]
            )
            ids = positions(rows, groups["discovery"])
            discovery_ids[corpus] = ids
            discovery_labels[corpus] = np.array(
                [rows[i]["is_genuine"] for i in ids], bool
            )
            del arrays
        for subset, candidate in candidates.items():
            path = run / "discovery" / subset_id(subset) / f"seed-{seed}"
            saved_report = read(path.with_suffix(".json"))
            assert sha(path.with_suffix(".npz")) == saved_report["score_sha256"]
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as scores:
                for corpus in config["corpora"]:
                    values = scores[corpus]
                    assert len(values) == len(discovery_ids[corpus])
                    assert np.isfinite(values).all()
                    value = independent_eer(values, discovery_labels[corpus])
                    error = abs(value - candidate["cells"][str(seed)][corpus])
                    assert error < 1e-12
                    assert (
                        saved_report["eers"][corpus]
                        == candidate["cells"][str(seed)][corpus]
                    )
                    max_error = max(max_error, error)
                    recomputed[subset].append(value)
                    total_discovery_scores += len(values)
        for split in ("validation", "test"):
            records = read(run / f"{split}-metrics.json")
            for corpus in config["corpora"]:
                arrays, data, rows = load_inputs(config, seed, corpus, split)
                ids = positions(rows, normal_only=False)
                labels = np.array([rows[i]["is_genuine"] for i in ids], bool)
                with np.load(
                    run / "scores" / f"{corpus}-{seed}-{split}.npz", allow_pickle=False
                ) as scores:
                    np.testing.assert_array_equal(scores["ids"], ids)
                    np.testing.assert_array_equal(scores["labels"], labels)
                    for variant in ("all36", "selected"):
                        values = scores[variant]
                        key = f"{corpus}/{seed}/{variant}"
                        threshold = selection["thresholds"][key]
                        numeric = (
                            np.inf
                            if threshold == "+inf"
                            else -np.inf
                            if threshold == "-inf"
                            else threshold
                        )
                        normal = np.array(
                            [rows[i]["role"] == "verification" for i in ids]
                        )
                        if split == "validation":
                            expected, _ = metrics.far_target_threshold(
                                values[normal], labels[normal], 0.01
                            )
                            assert metrics.json_threshold(expected) == threshold
                            subset = (
                                tuple(range(36))
                                if variant == "all36"
                                else tuple(winner["subset"])
                            )
                            path = (
                                run
                                / "discovery"
                                / subset_id(subset)
                                / f"seed-{seed}.npz"
                            )
                            indices = np.searchsorted(ids, discovery_ids[corpus])
                            np.testing.assert_array_equal(
                                ids[indices], discovery_ids[corpus]
                            )
                            with np.load(path, allow_pickle=False) as discovery:
                                np.testing.assert_array_equal(
                                    values[indices], discovery[corpus]
                                )
                        for row in records:
                            if (
                                row["seed"] != seed
                                or row["corpus"] != corpus
                                or row["variant"] != variant
                            ):
                                continue
                            members = set(
                                data["speakers"]
                                if row["cohort"] == "full"
                                else frozen["partitions"][corpus][row["cohort"]]
                            )
                            included = np.array(
                                [
                                    r["role"] == row["role"]
                                    and r["speaker_id"] in members
                                    and r["claimed_speaker_id"] in members
                                    for r in rows
                                ]
                            )
                            keep = included[ids]
                            y, x = labels[keep], values[keep]
                            original = [
                                r
                                for r, include in zip(rows, included, strict=True)
                                if include
                            ]
                            positive = sum(r["is_genuine"] for r in original)
                            negative = len(original) - positive
                            false_accepts = int(np.sum((x >= numeric) & ~y))
                            false_rejects = positive - int(np.sum((x >= numeric) & y))
                            expected = {
                                "eer": independent_eer(x, y),
                                "far": false_accepts / negative,
                                "frr": false_rejects / positive,
                                "coverage": int(y.sum()) / positive,
                                "false_accepts": false_accepts,
                                "false_rejects": false_rejects,
                                "all_genuine": positive,
                                "all_impostor": negative,
                                "scored_genuine": int(y.sum()),
                                "scored_impostor": int((~y).sum()),
                            }
                            for name, value in expected.items():
                                error = abs(value - row[name])
                                assert error < 1e-12, (key, name, value, row[name])
                                max_error = max(max_error, error)
                        total_evaluation_scores += len(values)
                del arrays
    for subset, values in recomputed.items():
        assert abs(np.mean(values) - candidates[subset]["objective"]) < 1e-12
    report = {
        "candidates": len(candidates),
        "search_replay_matches": True,
        "discovery_scores_recomputed": total_discovery_scores,
        "evaluation_scores_checked": total_evaluation_scores,
        "discovery_winner_scores_match_full_validation": True,
        "maximum_metric_difference": max_error,
        "files": pin(
            [
                BASE / "audit.py",
                run / "design-freeze.json",
                run / "winner-freeze.json",
                run / "selection-freeze.json",
                run / "complete-audit.json",
            ]
        ),
    }
    write(run / "independent-audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
