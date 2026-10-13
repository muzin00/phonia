"""Run a preregistered common-phone subset search on three frozen models."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import multiprocessing
import os
import time

from core import (
    BACKEND_PATH,
    BASE,
    BOOTSTRAP_PATH,
    METRICS_PATH,
    ROOT,
    Scorer,
    bootstrap,
    check,
    eer,
    independent_eer,
    load_inputs,
    load_model,
    metrics,
    np,
    partition,
    pin,
    positions,
    read,
    search,
    sha,
    subset_id,
    summarize,
    torch,
    write,
)

CONFIG = BASE / "config.json"
WORKER = {}


def state(run, status, **details):
    write(
        run / "state.json",
        {"status": status, "pid": os.getpid(), "time": time.time(), **details},
    )
    print(status, details, flush=True)


def model_path(config, seed):
    return (
        ROOT
        / config["source_run"]
        / f"acoustic-{seed}/continued"
        / f"seed-{config['fusion_seed']}/budget-{config['fusion_budget']}/selected.pt"
    )


def source_scores(config, seed, corpus, split):
    return (
        ROOT
        / config["source_run"]
        / "scores"
        / (
            f"{corpus}-src490-seed{seed}-u{config['fusion_budget']}-selected-{split}.json"
        )
    )


def freeze(config, run):
    destination = run / "design-freeze.json"
    if destination.exists():
        frozen = read(destination)
        assert frozen["config"] == config
        check(frozen["files"])
        return frozen
    source = ROOT / config["source_run"]
    audit = read(source / "complete-audit.json")
    check(audit["files"])
    check(read(source / "score-audit.json")["files"])
    phones = read(source / "design-freeze.json")["phones"]
    assert len(phones) == 36 and phones[:5] == ["a", "i", "u", "e", "o"]
    files = [
        CONFIG,
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        BASE / "README.md",
    ]
    files += [BACKEND_PATH, METRICS_PATH, BOOTSTRAP_PATH]
    files += [
        source / name
        for name in (
            "design-freeze.json",
            "complete-audit.json",
            "score-audit.json",
            "selection-freeze.json",
            "results.json",
        )
    ]
    partitions = {}
    for seed in config["encoder_seeds"]:
        family = source / f"acoustic-{seed}"
        files += [
            family / "trial/encoder.pt",
            family / "fusion/design-freeze.json",
            model_path(config, seed),
        ]
        check(read(family / "fusion/prepared-freeze.json")["files"])
        for corpus in config["corpora"]:
            for split in ("validation", "test"):
                check(
                    read(family / "fusion" / f"{corpus}-{split}-freeze.json")["files"]
                )
                files += [
                    family / "fusion" / f"{corpus}-{split}{suffix}"
                    for suffix in (".npz", "-inputs.json", "-baseline.json")
                ]
                files += [source_scores(config, seed, corpus, split)]
            data = read(family / "fusion" / f"{corpus}-validation-inputs.json")
            groups = partition(data["speakers"], config["partition_seed"])
            assert partitions.setdefault(corpus, groups) == groups
    frozen = {
        "config": config,
        "phones": phones,
        "partitions": partitions,
        "files": pin(files),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "created_at_epoch": time.time(),
        "all_test_previously_observed": True,
        "confirmation_already_used_for_source_checkpoint_selection": True,
    }
    write(destination, frozen, immutable=True)
    return frozen


def worker_init(config, frozen, seed):
    torch.set_num_threads(1)
    run = ROOT / config["run_directory"]
    model = load_model(config, seed)
    scorers, labels, reports = {}, {}, {}
    for corpus in config["corpora"]:
        arrays, _data, rows = load_inputs(config, seed, corpus, "validation")
        ids = positions(rows, frozen["partitions"][corpus]["discovery"])
        scorers[corpus] = Scorer(model, arrays, ids, config["batch_size"])
        labels[corpus] = np.array([rows[i]["is_genuine"] for i in ids], bool)
        reference = read(source_scores(config, seed, corpus, "validation"))
        values = scorers[corpus].score(tuple(range(36)))
        error = float(np.max(np.abs(values - [reference[i]["score"] for i in ids])))
        assert error < 1e-12, error
        # Independent full forward on several masks, including excluded vowels.
        probe = Scorer(model, arrays, ids[:256], config["batch_size"])
        errors = []
        for subset in (
            tuple(range(36)),
            tuple(range(1, 36)),
            (0, 5, 20, 22, 28),
            tuple(range(0, 36, 2)),
        ):
            errors.append(
                float(
                    np.max(
                        np.abs(probe.score(subset) - probe.score(subset, direct=True))
                    )
                )
            )
        assert max(errors) < 1e-12, errors
        reports[corpus] = {
            "discovery_trials": len(ids),
            "baseline_max_difference": error,
            "direct_forward_max_difference": max(errors),
        }
    write(run / f"preflight-{seed}.json", reports, immutable=True)
    WORKER.update(
        config=config, frozen=frozen, seed=seed, run=run, scorers=scorers, labels=labels
    )


def worker_evaluate(subsets):
    w = WORKER
    results = {}
    for subset in subsets:
        directory = w["run"] / "discovery" / subset_id(subset)
        path = directory / f"seed-{w['seed']}.json"
        score_path = directory / f"seed-{w['seed']}.npz"
        if path.exists():
            result = read(path)
            assert result["score_sha256"] == sha(score_path)
            assert result["subset"] == list(subset)
        else:
            values = {c: scorer.score(subset) for c, scorer in w["scorers"].items()}
            eers = {c: eer(v, w["labels"][c]) for c, v in values.items()}
            for c, v in values.items():
                assert abs(eers[c] - independent_eer(v, w["labels"][c])) < 1e-12
            directory.mkdir(parents=True, exist_ok=True)
            temp = score_path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **values)
            temp.replace(score_path)
            result = {
                "subset": list(subset),
                "seed": w["seed"],
                "eers": eers,
                "score_sha256": sha(score_path),
            }
            write(path, result, immutable=True)
        results[subset] = result["eers"]
    return results


def discover(config, run, frozen):
    destination = run / "winner-freeze.json"
    if destination.exists():
        winner = read(destination)
        assert winner["design_sha256"] == sha(run / "design-freeze.json")
        check(winner["discovery_files"])
        return winner
    ctx = multiprocessing.get_context("spawn")
    pools = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=worker_init,
            initargs=(config, frozen, seed),
        )
        for seed in config["encoder_seeds"]
    ]
    visited = {}

    def evaluate(subsets):
        unseen = [v for v in subsets if v not in visited]
        for start in range(0, len(unseen), 8):
            batch = unseen[start : start + 8]
            futures = [p.submit(worker_evaluate, batch) for p in pools]
            results = [f.result() for f in futures]
            for subset in batch:
                cells = {
                    str(seed): values[subset]
                    for seed, values in zip(
                        config["encoder_seeds"], results, strict=True
                    )
                }
                objective = float(
                    np.mean([v for item in cells.values() for v in item.values()])
                )
                visited[subset] = {
                    "subset": list(subset),
                    "objective": objective,
                    "cells": cells,
                }
            state(
                run,
                "searching",
                candidates=len(visited),
                current_batch_remaining=len(unseen) - start - len(batch),
            )
        return {v: visited[v]["objective"] for v in subsets}

    try:
        subset, history = search(
            frozen["phones"],
            evaluate,
            config,
            lambda h: write(run / "search-progress.json", h),
        )
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    write(
        run / "search-results.json",
        {"candidates": list(visited.values()), "history": history},
        immutable=True,
    )
    winner = {
        "subset": list(subset),
        "phones": [frozen["phones"][p] for p in subset],
        "removed": [p for i, p in enumerate(frozen["phones"]) if i not in subset],
        "objective": visited[subset]["objective"],
        "candidates": len(visited),
        "design_sha256": sha(run / "design-freeze.json"),
        "discovery_files": pin(
            [
                run / "search-results.json",
                *run.glob("preflight-*.json"),
                *run.glob("discovery/*/*"),
            ]
        ),
        "frozen_before_confirmation": True,
        "created_at_epoch": time.time(),
    }
    write(destination, winner, immutable=True)
    return winner


def numeric_threshold(value):
    return np.inf if value == "+inf" else -np.inf if value == "-inf" else value


def evaluate_split(config, run, frozen, winner, split, thresholds):
    if split == "test":
        selection = read(run / "selection-freeze.json")
        assert selection["winner_sha256"] == sha(run / "winner-freeze.json")
        assert selection["thresholds"] == thresholds
        check(selection["validation_files"])
    result, reports = [], []
    variants = {"all36": tuple(range(36)), "selected": tuple(winner["subset"])}
    for seed in config["encoder_seeds"]:
        model = load_model(config, seed)
        for corpus in config["corpora"]:
            state(run, "evaluating", split=split, seed=seed, corpus=corpus)
            arrays, data, rows = load_inputs(config, seed, corpus, split)
            ids = positions(rows, normal_only=False)
            labels = np.array([rows[i]["is_genuine"] for i in ids], bool)
            scorer = Scorer(model, arrays, ids, config["batch_size"])
            query_speakers = np.array(
                [data["speakers"].index(rows[i]["speaker_id"]) for i in ids]
            )
            claims = ids % len(data["speakers"])
            roles = np.array([rows[i]["role"] for i in ids])
            scores = {
                "ids": ids,
                "labels": labels,
                "query_speakers": query_speakers,
                "claims": claims,
                "roles": roles,
            }
            reference = read(source_scores(config, seed, corpus, split))
            for variant, subset in variants.items():
                values = scorer.score(subset)
                direct = scorer.score(subset, direct=True)
                error = float(np.max(np.abs(values - direct)))
                assert error < 1e-12, error
                if variant == "all36":
                    assert (
                        np.max(np.abs(values - [reference[i]["score"] for i in ids]))
                        < 1e-12
                    )
                scores[variant] = values
                normal = roles == "verification"
                key = f"{corpus}/{seed}/{variant}"
                if split == "validation":
                    point, _ = metrics.far_target_threshold(
                        values[normal], labels[normal], 0.01
                    )
                    thresholds[key] = metrics.json_threshold(point)
                threshold = numeric_threshold(thresholds[key])
                cohorts = {"full": set(data["speakers"])}
                if split == "validation":
                    cohorts.update(
                        {k: set(v) for k, v in frozen["partitions"][corpus].items()}
                    )
                for cohort, members in cohorts.items():
                    for role in sorted(set(roles)):
                        if cohort != "full" and role != "verification":
                            continue
                        keep = np.array(
                            [
                                rows[i]["role"] == role
                                and rows[i]["speaker_id"] in members
                                and rows[i]["claimed_speaker_id"] in members
                                for i in ids
                            ]
                        )
                        original = [
                            r
                            for r in rows
                            if r["role"] == role
                            and r["speaker_id"] in members
                            and r["claimed_speaker_id"] in members
                        ]
                        genuine_count = sum(r["is_genuine"] for r in original)
                        item = summarize(
                            values[keep],
                            labels[keep],
                            threshold,
                            genuine_count,
                            len(original) - genuine_count,
                        )
                        assert (
                            abs(
                                item["eer"]
                                - independent_eer(values[keep], labels[keep])
                            )
                            < 1e-12
                        )
                        if split == "test" and cohort == "full" and variant == "all36":
                            prior = read(ROOT / config["source_run"] / "results.json")[
                                "conditions"
                            ][corpus][
                                f"src490-seed{seed}-u{config['fusion_budget']}-selected"
                            ][role]
                            for name in ("eer", "far", "frr", "coverage"):
                                assert abs(item[name] - prior[name]) < 1e-12, (
                                    name,
                                    item[name],
                                    prior[name],
                                )
                        result.append(
                            {
                                "split": split,
                                "cohort": cohort,
                                "corpus": corpus,
                                "seed": seed,
                                "variant": variant,
                                "role": role,
                                **item,
                            }
                        )
                reports.append(
                    {
                        "split": split,
                        "corpus": corpus,
                        "seed": seed,
                        "variant": variant,
                        "records": len(ids),
                        "direct_forward_max_difference": error,
                    }
                )
            score_path = run / "scores" / f"{corpus}-{seed}-{split}.npz"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            if score_path.exists():
                with np.load(score_path, allow_pickle=False) as saved:
                    for name, value in scores.items():
                        np.testing.assert_array_equal(saved[name], value)
            else:
                temp = score_path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **scores)
                temp.replace(score_path)
            del scorer, arrays
    write(run / f"{split}-metrics.json", result, immutable=True)
    write(run / f"{split}-score-audit.json", reports, immutable=True)
    return result


def bootstrap_results(config, run):
    report, arrays = {}, {}
    for index, corpus in enumerate(config["corpora"]):
        differences = []
        for seed in config["encoder_seeds"]:
            with np.load(
                run / "scores" / f"{corpus}-{seed}-test.npz", allow_pickle=False
            ) as saved:
                normal = saved["roles"] == "verification"
                q, c = saved["query_speakers"][normal], saved["claims"][normal]
                _, counts = bootstrap.speaker_draws(
                    int(max(q.max(), c.max())) + 1,
                    config["bootstrap_replicates"],
                    config["bootstrap_seed"] + index,
                )
                estimates = {
                    v: bootstrap.weighted_eers(saved[v][normal], q, c, counts)
                    for v in ("all36", "selected")
                }
                # The independent weighted implementation must reproduce plain EER.
                ones = np.ones((1, counts.shape[1]), np.int64)
                for variant in estimates:
                    actual = bootstrap.weighted_eers(
                        saved[variant][normal], q, c, ones
                    )[0]
                    assert abs(actual - eer(saved[variant][normal], q == c)) < 1e-12
                delta = estimates["selected"] - estimates["all36"]
                arrays[f"{corpus}_{seed}"] = delta
                differences.append(delta)
        average = np.mean(differences, axis=0)
        arrays[corpus] = average
        report[corpus] = {
            "paired_eer_delta_ci95_pp": (
                100 * np.quantile(average, [0.025, 0.975])
            ).tolist()
        }
    average = np.mean([arrays[c] for c in config["corpora"]], axis=0)
    arrays["macro"] = average
    report["macro"] = {
        "paired_eer_delta_ci95_pp": (
            100 * np.quantile(average, [0.025, 0.975])
        ).tolist()
    }
    np.savez_compressed(run / "bootstrap.npz", **arrays)
    write(
        run / "bootstrap.json",
        {
            "method": "Paired two-axis speaker bootstrap. Same draws for both subsets and all acoustic seeds within corpus; corpus draws independent. Conditional on fixed models and selected subset; excludes selection/training uncertainty.",
            "replicates": config["bootstrap_replicates"],
            "conditions": report,
        },
        immutable=True,
    )
    return report


def report(config, run, frozen, winner):
    records = read(run / "validation-metrics.json") + read(run / "test-metrics.json")
    summary = []
    for split, cohort, role in (
        ("validation", "discovery", "verification"),
        ("validation", "confirmation", "verification"),
        ("validation", "full", "verification"),
        ("test", "full", "verification"),
        ("test", "full", "cross_text_verification"),
    ):
        for corpus in [*config["corpora"], "macro"]:
            chosen = [
                v
                for v in records
                if v["split"] == split
                and v["cohort"] == cohort
                and v["role"] == role
                and (corpus == "macro" or v["corpus"] == corpus)
            ]
            if not chosen or (role != "verification" and corpus == "macro"):
                continue
            result = {"split": split, "cohort": cohort, "role": role, "corpus": corpus}
            for variant in ("all36", "selected"):
                rows = [v for v in chosen if v["variant"] == variant]
                result[variant] = {
                    metric: float(np.mean([v[metric] for v in rows]))
                    for metric in ("eer", "far", "frr", "coverage")
                }
            deltas = []
            for seed in config["encoder_seeds"]:
                values = {
                    v: np.mean(
                        [
                            r["eer"]
                            for r in chosen
                            if r["seed"] == seed and r["variant"] == v
                        ]
                    )
                    for v in ("all36", "selected")
                }
                deltas.append(float(values["selected"] - values["all36"]))
            result.update(
                delta_eer_pp=100 * (result["selected"]["eer"] - result["all36"]["eer"]),
                seed_deltas_pp=[100 * v for v in deltas],
                improved_seeds=sum(v < -1e-12 for v in deltas),
                worsened_seeds=sum(v > 1e-12 for v in deltas),
            )
            summary.append(result)
    confidence = read(run / "bootstrap.json")
    payload = {
        "config": config,
        "winner": winner["phones"],
        "removed": winner["removed"],
        "candidates": winner["candidates"],
        "summary": summary,
        "conditions": records,
        "bootstrap": confidence,
    }
    write(run / "results.json", payload, immutable=True)
    write(BASE / "evaluation-results.json", payload)
    with (run / "conditions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# 固定SRC490モデルの音素選別",
        "",
        f"選択: {' / '.join(winner['phones'])}（{len(winner['phones'])}種類）。",
        f"除外: {' / '.join(winner['removed']) or 'なし'}。探索した異なる集合は{winner['candidates']}通り。",
        "",
        "3音響seed平均。EERは採点可能分、FAR/FRRは未採点を拒否に含む全入力。平均差はselected−all36、単位はパーセントポイント。",
        "",
        "| 集合 | コーパス | 全36 EER | 選択後 EER | 差 | 改善 / 悪化 seed |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        if row["role"] != "verification":
            continue
        label = f"{row['split']}/{row['cohort']}"
        lines.append(
            f"| {label} | {row['corpus']} | {100 * row['all36']['eer']:.3f}% | {100 * row['selected']['eer']:.3f}% | {row['delta_eer_pp']:+.3f} | {row['improved_seeds']} / {row['worsened_seeds']} |"
        )
    lines += [
        "",
        "## 固定validation FAR1%閾値でのtest",
        "",
        "| コーパス | FAR 全36→選択 | FRR 全36→選択 | 採点率 | EER差の対応付き95%区間 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        if row["split"] != "test" or row["role"] != "verification":
            continue
        a, b = row["all36"], row["selected"]
        lo, hi = confidence["conditions"][row["corpus"]]["paired_eer_delta_ci95_pp"]
        lines.append(
            f"| {row['corpus']} | {100 * a['far']:.3f}→{100 * b['far']:.3f}% | {100 * a['frr']:.3f}→{100 * b['frr']:.3f}% | {100 * a['coverage']:.3f}% | [{lo:+.3f}, {hi:+.3f}] pp |"
        )
    lines += [
        "",
        "## 採用までの探索",
        "",
        "| 操作 | 除外 | 追加 | 音素数 | 探索EER |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in read(run / "search-results.json")["history"]:
        lines.append(
            f"| {row['move']} | {' / '.join(row.get('removed', []))} | {' / '.join(row.get('added', []))} | {len(row['subset'])} | {100 * row['objective']:.3f}% |"
        )
    lines += [
        "",
        "## seed別test EER",
        "",
        "| コーパス | 音響seed | 全36 | 選択後 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for corpus in config["corpora"]:
        for seed in config["encoder_seeds"]:
            values = {
                v["variant"]: v["eer"]
                for v in records
                if v["split"] == "test"
                and v["role"] == "verification"
                and v["corpus"] == corpus
                and v["seed"] == seed
            }
            lines.append(
                f"| {corpus} | {seed} | {100 * values['all36']:.3f}% | {100 * values['selected']:.3f}% |"
            )
    lines += [
        "",
        "## 解釈の範囲",
        "",
        "探索話者と確認話者はquery・claimed speakerの両方で分離した。確認用話者は音素探索には未使用だが、元モデルのcheckpoint選択には使用済みであり、完全に独立したholdoutではない。確認結果で選択を変更せず、test前に集合・全validation閾値を固定した。",
        "",
        "testも既観測の同じ話者集合。3音響seedは新規母集団ではない。95%区間は固定モデル・固定集合を条件とし、探索や学習の変動を含まない。母音も除外対象だが、少なくとも1母音を残し、全条件で従来の5母音必須ゲートを維持した。採点率改善は今回の対象外。",
        "",
        "Transformerのattention・softmaxを毎回再計算した。保存済み重みの再正規化だけではない。選別で実使用音声量も減るため、固定総時間の比較ではない。探索は上限付きの局所探索で、最適集合や音素固有の普遍的な優劣を確定しない。再学習は未実施。",
        "",
        f"成果物: `{config['run_directory']}`。再現条件は[README](README.md)と[config.json](config.json)。",
        "",
    ]
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)


def execute(config, run):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        state(run, "freezing")
        frozen = freeze(config, run)
        if (run / "complete-audit.json").exists():
            check(read(run / "complete-audit.json")["files"])
            state(run, "complete", verified_existing_run=True)
            return
        winner = discover(config, run, frozen)
        state(
            run,
            "winner_frozen",
            phones=winner["phones"],
            candidates=winner["candidates"],
        )
        thresholds = {}
        evaluate_split(config, run, frozen, winner, "validation", thresholds)
        write(
            run / "selection-freeze.json",
            {
                "winner_sha256": sha(run / "winner-freeze.json"),
                "thresholds": thresholds,
                "validation_files": pin(
                    [
                        *run.glob("scores/*-validation.npz"),
                        run / "validation-metrics.json",
                        run / "validation-score-audit.json",
                    ]
                ),
            },
            immutable=True,
        )
        evaluate_split(config, run, frozen, winner, "test", thresholds)
        state(run, "bootstrap")
        bootstrap_results(config, run)
        report(config, run, frozen, winner)
        check(frozen["files"])
        check(winner["discovery_files"])
        files = [
            p
            for p in run.rglob("*")
            if p.is_file()
            and p.name not in ("state.json", "execution.lock", "complete-audit.json")
            and not p.name.endswith(".tmp")
        ]
        write(
            run / "complete-audit.json",
            {
                "source_files_unchanged": True,
                "discovery_cache_unchanged": True,
                "all36_matches_source_scores": True,
                "all36_matches_source_test_metrics": True,
                "cached_projection_matches_direct_forward": True,
                "same_scored_trials_all_candidates": True,
                "files": pin(files),
            },
            immutable=True,
        )
        state(run, "complete", artifact_directory=str(run.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["execute", "status"])
    args = parser.parse_args()
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    if args.command == "status":
        print(read(run / "state.json"))
    else:
        try:
            execute(config, run)
        except BaseException as error:
            state(run, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
