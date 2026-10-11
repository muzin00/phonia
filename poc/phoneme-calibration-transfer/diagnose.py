"""Frozen validation thresholds and explicitly retrospective FAR-matched curves."""

from __future__ import annotations

import argparse
import fcntl
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-corpus-seed-replication"))
import replicate as r

s = r.s
CONFIG = BASE / "config.json"


def key(target):
    return f"{target:.6f}"


def normal(records):
    return [row for row in records if row["role"] == "verification"]


def scored_arrays(records):
    scored = [row for row in records if row["score"] is not None]
    return s.np.array([row["score"] for row in scored]), s.np.array(
        [row["is_genuine"] for row in scored], dtype=bool
    )


def empirical_cap_threshold(records, cap):
    """Lowest representable threshold respecting an all-input impostor FAR cap.

    This uses test labels only for retrospective ROC diagnosis, never deployment.
    Missing impostors count in the denominator and are always rejected.
    """
    assert 0 <= cap <= 1
    impostors = [row for row in records if not row["is_genuine"]]
    if not impostors:
        raise ValueError("no impostor trials")
    values = sorted(row["score"] for row in impostors if row["score"] is not None)
    assert all(math.isfinite(value) for value in values)
    allowed = math.floor(cap * len(impostors))
    if allowed >= len(values):
        return -math.inf
    boundary = values[len(values) - allowed - 1]
    return float(s.np.nextafter(boundary, s.np.inf))


def distribution(records, quantiles):
    scores, labels = scored_arrays(records)
    return {
        name: {
            "scored": int(mask.sum()),
            "quantiles": dict(
                zip(
                    map(str, quantiles),
                    s.np.quantile(scores[mask], quantiles).tolist(),
                    strict=True,
                )
            ),
        }
        for name, mask in (("genuine", labels), ("impostor", ~labels))
    }


def initialize(config, run):
    source = s.ROOT / config["source_run"]
    assert s.read(source / "execution-state.json")["status"] == "complete"
    audit = s.read(source / "completion-audit.json")
    assert audit["status"] == "passed"
    r.checked_files(audit["files"])
    evaluation = s.read(source / "evaluation-freeze.json")
    r.checked_files(evaluation["validation_files"])
    paths = [CONFIG, Path(__file__), *sorted((BASE / "tests").glob("*.py"))]
    paths += [
        source / name
        for name in (
            "design-freeze.json",
            "completion-audit.json",
            "evaluation-freeze.json",
            "all-models-freeze.json",
            "results.json",
        )
    ]
    for seed in config["seeds"]:
        directory = source / f"seed-{seed}"
        paths += [directory / "results.json"]
        paths += [
            directory / "scores" / f"{corpus}-{arm}-{split}.json"
            for corpus in s.CORPORA
            for arm in config["arms"]
            for split in ("validation", "test")
        ]
    frozen = {
        "config": config,
        "files": r.pin(paths),
        "runtime": s.expanded.runtime(),
        "all_test_previously_observed": True,
    }
    s.fixed(run / "design-freeze.json", frozen)
    return evaluation


def load(config, seed, corpus, arm, split):
    path = (
        s.ROOT
        / config["source_run"]
        / f"seed-{seed}"
        / "scores"
        / f"{corpus}-{arm}-{split}.json"
    )
    return normal(s.read(path))


def calibrate(config, run, original):
    thresholds, stats = {}, {}
    for seed in config["seeds"]:
        thresholds[str(seed)], stats[str(seed)] = {}, {}
        for corpus in s.CORPORA:
            thresholds[str(seed)][corpus], stats[str(seed)][corpus] = {}, {}
            for arm in config["arms"]:
                records = load(config, seed, corpus, arm, "validation")
                scores, labels = scored_arrays(records)
                points, metrics = {}, {}
                for target in config["validation_scored_far_targets"]:
                    value, _ = s.far_target_threshold(scores, labels, target)
                    point = s.json_threshold(value)
                    points[key(target)] = point
                    metrics[key(target)] = s.operating(records, point)
                assert (
                    points[key(0.01)]
                    == original["seeds"][str(seed)]["thresholds"][corpus][arm]
                )
                thresholds[str(seed)][corpus][arm] = points
                stats[str(seed)][corpus][arm] = {
                    "operating": metrics,
                    "distribution": distribution(records, config["quantiles"]),
                }
    s.fixed(
        run / "validation-freeze.json",
        {
            "thresholds": thresholds,
            "validation": stats,
            "test_used_to_choose_thresholds": False,
            "design_sha256": s.sha(run / "design-freeze.json"),
        },
    )
    return thresholds, stats


def analyze(config, run, thresholds, validation):
    rows = {}
    source = s.ROOT / config["source_run"]
    for seed in config["seeds"]:
        rows[str(seed)] = {}
        prior = s.read(source / f"seed-{seed}" / "results.json")
        for corpus in s.CORPORA:
            rows[str(seed)][corpus] = {}
            expected_keys = expected_missing = None
            for arm in config["arms"]:
                records = load(config, seed, corpus, arm, "test")
                identities = [
                    (
                        v["query_id"],
                        v["speaker_id"],
                        v["claimed_speaker_id"],
                        v["is_genuine"],
                    )
                    for v in records
                ]
                missing = [v["score"] is None for v in records]
                if expected_keys is None:
                    expected_keys, expected_missing = identities, missing
                assert identities == expected_keys and missing == expected_missing
                locked, roc = {}, {}
                for target in config["validation_scored_far_targets"]:
                    point = thresholds[str(seed)][corpus][arm][key(target)]
                    test = s.operating(records, point)
                    val = validation[str(seed)][corpus][arm]["operating"][key(target)]
                    locked[key(target)] = {
                        "test": test,
                        "validation": val,
                        "scored_far_transfer_gap": test["scored_only_error_rates"][
                            "far"
                        ]
                        - val["scored_only_error_rates"]["far"],
                        "all_input_far_transfer_gap": test["far"] - val["far"],
                    }
                assert (
                    locked[key(0.01)]["test"]
                    == prior["conditions"][corpus][arm]["verification"]
                )
                for cap in config["diagnostic_test_all_input_far_caps"]:
                    point = s.json_threshold(empirical_cap_threshold(records, cap))
                    result = s.operating(records, point)
                    assert result["far"] <= cap
                    roc[key(cap)] = {
                        "diagnostic_only": True,
                        "test_labels_used_for_roc": True,
                        "not_a_deployment_threshold": True,
                        "operating": result,
                    }
                rows[str(seed)][corpus][arm] = {
                    "validation_locked": locked,
                    "retrospective_roc": roc,
                    "test_distribution": distribution(records, config["quantiles"]),
                }
    s.fixed(run / "per-seed-results.json", rows)
    return rows


def summarize(values, differences):
    values, differences = s.np.asarray(values), s.np.asarray(differences)
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "paired_delta_vs_expanded280": differences.tolist(),
        "paired_delta_mean": float(differences.mean()),
        "decreased_seeds": int((differences < 0).sum()),
        "increased_seeds": int((differences > 0).sum()),
    }


def aggregate(config, run, rows):
    modes = {
        "validation_locked": config["validation_scored_far_targets"],
        "retrospective_roc": config["diagnostic_test_all_input_far_caps"],
    }
    result = {}
    for mode, targets in modes.items():
        result[mode] = {}
        for target in targets:
            result[mode][key(target)] = {}
            for corpus in s.CORPORA:
                result[mode][key(target)][corpus] = {}
                for arm in config["arms"]:
                    metrics = {}
                    for metric in ("far", "frr", "coverage"):
                        values, delta = [], []
                        for seed in config["seeds"]:
                            source = rows[str(seed)][corpus]
                            field = (
                                "test" if mode == "validation_locked" else "operating"
                            )
                            value = source[arm][mode][key(target)][field][metric]
                            base = source["expanded280"][mode][key(target)][field][
                                metric
                            ]
                            values.append(value)
                            delta.append(value - base)
                        metrics[metric] = summarize(values, delta)
                    if mode == "validation_locked":
                        metrics["scored_far_transfer_gap"] = [
                            rows[str(seed)][corpus][arm][mode][key(target)][
                                "scored_far_transfer_gap"
                            ]
                            for seed in config["seeds"]
                        ]
                    result[mode][key(target)][corpus][arm] = metrics
    report = {
        "aggregate": result,
        "config": config,
        "validation_freeze_sha256": s.sha(run / "validation-freeze.json"),
        "per_seed_sha256": s.sha(run / "per-seed-results.json"),
        "no_operational_threshold_selected": True,
    }
    s.fixed(run / "results.json", report)
    lines = [
        "# 校正移行とFARを揃えた事後診断",
        "",
        "5seed平均。校正閾値はvalidationだけから固定。ROC診断はtest上の事後比較であり運用閾値ではない。",
        "",
    ]
    for mode in modes:
        lines += [
            f"## {mode}",
            "",
            "| 目標/上限FAR | コーパス | 条件 | test FAR | 本人拒否率 | 元の280との差(pt) | 本人拒否低下seed |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for target, corpora in result[mode].items():
            for corpus, arms in corpora.items():
                for arm, metrics in arms.items():
                    lines.append(
                        f"| {float(target) * 100:.1f}% | {corpus} | {arm} | {metrics['far']['mean'] * 100:.3f}% | {metrics['frr']['mean'] * 100:.3f}% | {metrics['frr']['paired_delta_mean'] * 100:+.3f} | {metrics['frr']['decreased_seeds']}/5 |"
                    )
        lines.append("")
    lines += [
        "validation_lockedの目標は採点可能な校正他人試行を分母とする。retrospective_rocの上限は未採点を含む全test他人試行を分母とし、同点を分割せず離散的な上限以下の点を使う。",
        "全testは既観測。seed間の標本SDは独立の人物母集団の信頼区間ではない。未採点は拒否に含む。",
    ]
    (run / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "execute"))
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = s.read(CONFIG)
    run = s.ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            original = initialize(config, run)
            if args.phase == "execute":
                s.atomic(run / "execution-state.json", {"status": "calibrating"})
                points, validation = calibrate(config, run, original)
                s.atomic(run / "execution-state.json", {"status": "analyzing"})
                rows = analyze(config, run, points, validation)
                aggregate(config, run, rows)
                s.atomic(
                    run / "execution-state.json",
                    {
                        "status": "complete",
                        "no_model_or_operational_threshold_changed": True,
                    },
                )
        except BaseException as error:
            s.atomic(
                run / "execution-state.json", {"status": "failed", "error": str(error)}
            )
            raise


if __name__ == "__main__":
    main()
