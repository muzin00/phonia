"""Aggregate exploratory results, paired uncertainty and independent replay checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import study as t

s, np, a = t.s, t.np, t.a


def interval(values):
    values = np.asarray(values)
    return np.quantile(values[np.isfinite(values)], [0.025, 0.975]).tolist()


def rates(values, threshold, speakers, counts):
    lookup = {sp: i for i, sp in enumerate(speakers)}
    n = len(speakers)
    q = np.array([lookup[r["speaker_id"]] for r in values])
    c = np.array([lookup[r["claimed_speaker_id"]] for r in values])
    genuine = q == c
    accepted = np.array(
        [r["status"] == "scored" and r["score"] >= float(threshold) for r in values]
    )
    weights = (counts[:, :, None] * counts[:, None, :]).reshape(len(counts), -1)
    weights[:, np.arange(n) * (n + 1)] = counts

    def total(mask):
        bins = np.bincount((q * n + c)[mask], minlength=n * n)
        return weights @ bins

    with np.errstate(invalid="ignore", divide="ignore"):
        draws = {
            "far": total(~genuine & accepted) / total(~genuine),
            "frr": total(genuine & ~accepted) / total(genuine),
        }
    # Direct summation independently checks clustered weights on 20 draws.
    for b in range(0, len(counts), max(1, len(counts) // 20)):
        w = np.where(genuine, counts[b, q], counts[b, q] * counts[b, c])
        for key, denominator, numerator in [
            ("far", ~genuine, ~genuine & accepted),
            ("frr", genuine, genuine & ~accepted),
        ]:
            if w[denominator].sum():
                assert (
                    abs(w[numerator].sum() / w[denominator].sum() - draws[key][b])
                    < 1e-12
                )
    return draws


def metrics(values, threshold, speakers, counts):
    scored = [r for r in values if r["status"] == "scored"]
    total_g = sum(r["is_genuine"] for r in values)
    total_i = len(values) - total_g
    g = np.array([r["score"] for r in scored if r["is_genuine"]])
    i = np.array([r["score"] for r in scored if not r["is_genuine"]])
    fa = int(np.sum(i >= float(threshold)))
    fr = total_g - int(np.sum(g >= float(threshold)))
    scores = np.array([r["score"] for r in scored])
    genuine = np.array([r["is_genuine"] for r in scored])
    eer = t.h.roc_eer(scores, genuine)["eer"] if len(g) and len(i) else None
    if eer is not None:
        assert abs(eer - s.original.independent_eer(scores, genuine)) < 1e-12
    draws = rates(values, threshold, speakers, counts)
    return {
        "queries": total_g,
        "impostor_trials": total_i,
        "speakers": len({r["speaker_id"] for r in values}),
        "scored_queries": len(g),
        "coverage": len(g) / total_g,
        "eer": eer,
        "threshold": threshold,
        "false_accepts": fa,
        "false_rejects": fr,
        "far": fa / total_i,
        "frr": fr / total_g,
        "far_ci95": interval(draws["far"]),
        "frr_ci95": interval(draws["frr"]),
        "genuine_score_quantiles": np.quantile(g, [0.1, 0.5, 0.9]).tolist()
        if len(g)
        else [],
        "impostor_score_quantiles": np.quantile(i, [0.1, 0.5, 0.9]).tolist()
        if len(i)
        else [],
    }, draws


def counts_for(speakers, config):
    rng = np.random.default_rng(config["bootstrap_seed"])
    n = len(speakers)
    return rng.multinomial(n, np.ones(n) / n, config["bootstrap_replicates"])


def summarize_quality(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row["corpus"], row["split"], row["role"]].append(row)
    result = []
    fields = list(a.quality(np.zeros(480)))
    for (corpus, split, role), group in sorted(groups.items()):
        stats = {
            m: {
                "p10": float(np.quantile([r[m] for r in group], 0.1)),
                "median": float(np.median([r[m] for r in group])),
                "p90": float(np.quantile([r[m] for r in group], 0.9)),
            }
            for m in fields
        }
        result.append(
            {
                "corpus": corpus,
                "split": split,
                "role": role,
                "recordings": len(group),
                "speakers": len({r["speaker_id"] for r in group}),
                "recordings_with_near_clipping": sum(
                    r["near_clip_fraction"] > 0 for r in group
                ),
                "metrics": stats,
            }
        )
    # Corpus differences are resampled by speaker, retaining each speaker's recordings.
    comparisons = []
    for split, role in [
        ("train", "training"),
        ("validation", "verification"),
        ("test", "verification"),
    ]:
        per_corpus = {}
        for corpus in ("JVS", "CV"):
            group = groups[corpus, split, role]
            speakers = sorted({r["speaker_id"] for r in group})
            rng = np.random.default_rng(a.seed_for(config["bootstrap_seed"], corpus))
            draws = rng.integers(
                0, len(speakers), size=(config["bootstrap_replicates"], len(speakers))
            )
            per_corpus[corpus] = {}
            for field in fields:
                medians = np.array(
                    [
                        np.median([r[field] for r in group if r["speaker_id"] == sp])
                        for sp in speakers
                    ]
                )
                per_corpus[corpus][field] = {
                    "estimate": float(np.mean(medians)),
                    "draws": medians[draws].mean(axis=1),
                }
        comparisons.append(
            {
                "split": split,
                "role": role,
                "estimand": "CV minus JVS mean of within-speaker medians",
                "metrics": {
                    field: {
                        "difference": per_corpus["CV"][field]["estimate"]
                        - per_corpus["JVS"][field]["estimate"],
                        "ci95": interval(
                            per_corpus["CV"][field]["draws"]
                            - per_corpus["JVS"][field]["draws"]
                        ),
                    }
                    for field in fields
                },
            }
        )
    return result, comparisons


def quality_strata(config, run, rows):
    boundaries = s.read_json(run / "quality-strata-freeze.json")["boundaries"]
    points = s.read_json(t.ROOT / config["cv_run"] / "threshold-freeze.json")[
        "thresholds"
    ]
    old_points = s.read_json(t.ROOT / config["missing_run"] / "threshold-freeze.json")[
        "thresholds"
    ]["equal"]
    values = list(s.rows(t.ROOT / config["cv_run"] / "test/candidates.jsonl"))
    speakers = sorted({r["speaker_id"] for r in values})
    counts = counts_for(speakers, config)
    quality = {
        r["query_id"]: r
        for r in rows
        if r["corpus"] == "CV" and r["split"] == "test" and r["role"] == "verification"
    }
    result = []
    draws_by_group = {}
    for field, limits in boundaries.items():
        groups = {
            label: {
                qid
                for qid, r in quality.items()
                if int(np.searchsorted(limits, r[field], side="right")) == bin_number
            }
            for bin_number, label in enumerate(("low", "middle", "high"))
        }
        assert len(set.union(*groups.values())) == 900
        for label, identifiers in groups.items():
            subset = [r for r in values if r["query_id"] in identifiers]
            for policy in config["policies"]:
                chosen = s.apply_policy(subset, policy)
                primary, draws = metrics(
                    chosen, points[policy["name"]]["far_1pct"], speakers, counts
                )
                transferred, _ = metrics(
                    chosen, old_points[policy["name"]]["far_1pct"], speakers, counts
                )
                draws_by_group[field, label, policy["name"]] = draws
                result.append(
                    {
                        "field": field,
                        "stratum": label,
                        "policy": policy["name"],
                        "boundaries": limits,
                        "active_seconds_median": float(
                            np.median(
                                [quality[q]["active_seconds"] for q in identifiers]
                            )
                        ),
                        "quality_median": float(
                            np.median([quality[q][field] for q in identifiers])
                        ),
                        "cv_calibrated": primary,
                        "jvs_transferred": transferred,
                    }
                )
    differences = []
    for field in boundaries:
        for policy in config["policies"]:
            low = draws_by_group[field, "low", policy["name"]]
            high = draws_by_group[field, "high", policy["name"]]
            differences.append(
                {
                    "field": field,
                    "policy": policy["name"],
                    "contrast": "high minus low",
                    "far_difference_ci95": interval(high["far"] - low["far"]),
                    "frr_difference_ci95": interval(high["frr"] - low["frr"]),
                }
            )
    # Descriptive crossed strata; residual speaker/phoneme confounding remains.
    crossed = []
    contrast_mid = np.median(
        [
            r["energy_contrast_db"]
            for r in rows
            if r["corpus"] == "CV"
            and r["split"] == "validation"
            and r["role"] == "verification"
        ]
    )
    duration_mid = np.median(
        [
            r["active_seconds"]
            for r in rows
            if r["corpus"] == "CV"
            and r["split"] == "validation"
            and r["role"] == "verification"
        ]
    )
    for noise_label in ("low", "high"):
        for length_label in ("short", "long"):
            ids = {
                q
                for q, r in quality.items()
                if ("low" if r["energy_contrast_db"] < contrast_mid else "high")
                == noise_label
                and ("short" if r["active_seconds"] < duration_mid else "long")
                == length_label
            }
            for policy in config["policies"]:
                chosen = s.apply_policy(
                    [r for r in values if r["query_id"] in ids], policy
                )
                report, _ = metrics(
                    chosen, points[policy["name"]]["far_1pct"], speakers, counts
                )
                crossed.append(
                    {
                        "contrast": noise_label,
                        "duration": length_label,
                        "policy": policy["name"],
                        "contrast_boundary": float(contrast_mid),
                        "duration_boundary": float(duration_mid),
                        "metrics": report,
                    }
                )
    return result, differences, crossed


def stress_report(config, run):
    calibration = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", calibration["design_sha256"])
    for path, digest in calibration["files"].items():
        s.checked(t.ROOT / path, digest)
    old_points = s.read_json(t.ROOT / config["missing_run"] / "threshold-freeze.json")[
        "thresholds"
    ]["equal"]
    speakers = t.jvs_data(config, "test")["speakers"]
    counts = counts_for(speakers, config)
    reports, draws = [], {}
    for condition in config["conditions"]:
        diagnostics = s.read_json(run / "test" / condition / "diagnostics.json")
        shifts = [v for r in diagnostics for v in r["matched_crop_shift_seconds"]]
        extra = {
            "alignment_failures": sum(
                r["alignment_failure"] is not None for r in diagnostics
            ),
            "crop_shift_median_seconds": float(np.median(shifts)),
            "crop_shift_p90_seconds": float(np.quantile(shifts, 0.9)),
            "fixed_intervals": sum(r["fixed_intervals"] for r in diagnostics),
            "realigned_intervals": sum(r["realigned_intervals"] for r in diagnostics),
            "maximum_clip_fraction": max(
                r["transform"]["pre_quantization_clip_fraction"] for r in diagnostics
            ),
            "quality_medians": {
                field: float(np.median([r["quality"][field] for r in diagnostics]))
                for field in diagnostics[0]["quality"]
            },
        }
        for mode in config["modes"]:
            raw = list(s.rows(run / "test" / condition / f"{mode}-scores.jsonl"))
            for policy in config["policies"]:
                values = s.apply_policy(raw, policy)
                original, draw = metrics(
                    values, old_points[policy["name"]]["far_1pct"], speakers, counts
                )
                calibrated, _ = metrics(
                    values,
                    calibration["thresholds"][f"{condition}/{mode}"][policy["name"]][
                        "far_1pct"
                    ],
                    speakers,
                    counts,
                )
                draws[condition, mode, policy["name"]] = draw
                clean = draws["clean", mode, policy["name"]]
                difference = {
                    key: interval(draw[key] - clean[key]) for key in ("far", "frr")
                }
                fixed = draws[condition, "fixed", policy["name"]]
                extra_difference = {
                    key: interval(draw[key] - fixed[key]) for key in ("far", "frr")
                }
                reports.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "policy": policy["name"],
                        "original_threshold": original,
                        "condition_calibrated": calibrated,
                        "difference_from_clean_ci95": difference,
                        "difference_from_fixed_ci95": extra_difference,
                        **extra,
                    }
                )
    return reports


def audit(config, run):
    design = t.verify(run)
    checked_rows = 0
    maximum = 0.0
    model, _ = s.original.load_model(
        t.ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
    )
    pipe = t.h.primitives.pipeline(design["extract_config"])
    for split in ("validation", "test"):
        data = t.jvs_data(config, split)
        clean_vectors = np.load(
            t.ROOT / config["jvs_run"] / f"{split}-embeddings.npy", allow_pickle=False
        ).astype(np.float64)
        registration = t.profiles(data, clean_vectors)
        source_values = {
            (r["query_id"], r["claimed_speaker_id"]): r
            for r in s.rows(
                t.ROOT / config["missing_run"] / split / "equal-candidates.jsonl"
            )
        }
        for condition in config["conditions"]:
            directory = run / split / condition
            references = {m: [] for m in config["modes"]}
            for number, path in enumerate(sorted(directory.glob("*/prepared.json"))):
                prepared = s.read_json(path)
                for name, digest in prepared["files"].items():
                    s.checked(t.ROOT / name, digest)
                for name, digest in s.read_json(path.parent / "cache.json").items():
                    s.checked(t.ROOT / name, digest)
                with np.load(path.parent / "vectors.npz", allow_pickle=False) as z:
                    vectors = {
                        m: z[m].astype(np.float64) for m in ("fixed", "realigned")
                    }
                for mode in config["modes"]:
                    source = "fixed" if mode == "fixed_qc" else mode
                    references[mode].extend(
                        t.score(
                            prepared["query"],
                            data,
                            vectors[source],
                            prepared[source],
                            registration,
                            mode,
                            split,
                        )
                    )
                if number == 0:
                    y = a.read_wave(t.ROOT / prepared["metadata"]["source_file"])
                    original_source = next(
                        item["metadata"]["source_file"]
                        for item in s.read_json(run / "selection.json")["stress"][split]
                        if item["query"]["query_id"] == prepared["query"]["query_id"]
                    )
                    reconstructed, _ = a.transform(
                        a.read_wave(t.ROOT / original_source),
                        condition,
                        a.seed_for(config["noise_seed"], prepared["query"]["query_id"]),
                    )
                    np.testing.assert_array_equal(y, reconstructed)
                    with np.load(path.parent / "features.npz", allow_pickle=False) as z:
                        for mode in ("fixed", "realigned"):
                            if len(z[mode]):
                                recomputed = s.original.model_embeddings(model, z[mode])
                                np.testing.assert_array_equal(
                                    recomputed.astype(np.float64), vectors[mode]
                                )
                                r = prepared[mode][0]
                                f = t.h.primitives.pool_features(
                                    pipe,
                                    y[r["start_frame"] : r["end_frame"]],
                                    r["segment_id"],
                                )
                                np.testing.assert_array_equal(f, z[mode][0])
            for mode in config["modes"]:
                by_pair = {
                    (r["query_id"], r["claimed_speaker_id"]): r
                    for r in references[mode]
                }
                actual = list(s.rows(directory / f"{mode}-scores.jsonl"))
                for row in actual:
                    key = row["query_id"], row["claimed_speaker_id"]
                    reference = by_pair[key]
                    assert row["used_phones"] == reference["used_phones"]
                    assert row["status"] == reference["status"]
                    if row["score"] is not None:
                        maximum = max(maximum, abs(row["score"] - reference["score"]))
                        if condition == "clean" and mode == "fixed":
                            assert (
                                abs(row["score"] - source_values[key]["score"]) < 2e-6
                            )
                    checked_rows += 1
                assert len(actual) == len(by_pair) == len(data["speakers"]) * 150
    assert maximum < 2e-6
    return {
        "status": "passed",
        "independent_float64_score_rows": checked_rows,
        "max_score_error": maximum,
        "model_and_pcm_replay_cases": 12,
        "bootstrap_check": "20 direct weighted-count checks per rate estimate",
        "scipy_version": __import__("scipy").__version__,
    }


def markdown(results):
    lines = [
        "# 録音品質と照合性能の診断結果",
        "",
        "既に観測済みの話者を用いた探索的診断。FAR/FRRは採点不能を拒否に含む全入力割合、EERは採点可能入力のみ。単位は%。",
        "",
        "## コーパスの品質指標",
        "",
        "中央値。contrastは真のSNRではなく、20msフレームのRMS P90−P10。",
        "",
        "| 集合 | コーパス | 音声数 | contrast dB | active RMS dBFS | quiet RMS dBFS | active秒 | 95%帯域Hz | clip近傍あり件数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results["quality_summary"]:
        if r["role"] == "enrollment":
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['split']}/{r['role']} | {r['corpus']} | {r['recordings']} | {m['energy_contrast_db']['median']:.2f} | {m['active_rms_dbfs']['median']:.2f} | {m['quiet_rms_dbfs']['median']:.2f} | {m['active_seconds']['median']:.2f} | {m['rolloff95_hz']['median']:.0f} | {r['recordings_with_near_clipping']} |"
        )
    lines += [
        "",
        "## 同一JVS音声への加工",
        "",
        "各条件test150発話・本人150試行/他人2,100試行。登録clean固定。元JVS閾値と加工条件別validation校正を比較。",
        "",
        "| 加工 | 境界/QC | 受付 | coverage | EER | 元閾値FAR | 元閾値FRR | 条件校正FAR | 条件校正FRR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results["stress"]:
        m, n = r["original_threshold"], r["condition_calibrated"]
        eer = f"{100 * m['eer']:.3f}" if m["eer"] is not None else "—"
        lines.append(
            f"| {r['condition']} | {r['mode']} | {r['policy']} | {100 * m['coverage']:.3f} | {eer} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} | {100 * n['far']:.3f} | {100 * n['frr']:.3f} |"
        )
    lines += [
        "",
        "## CV testの品質別比較",
        "",
        "境界はCV validationの三分位。閾値は前回CV校正の固定値。品質以外に話者・文長・音素構成も異なる。",
        "",
        "| 指標 | 層 | 受付 | 発話/話者 | active秒中央値 | coverage | EER | FAR | FRR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results["cv_strata"]:
        m = r["cv_calibrated"]
        eer = f"{100 * m['eer']:.3f}" if m["eer"] is not None else "—"
        lines.append(
            f"| {r['field']} | {r['stratum']} | {r['policy']} | {m['queries']}/{m['speakers']} | {r['active_seconds_median']:.2f} | {100 * m['coverage']:.3f} | {eer} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} |"
        )
    lines += [
        "",
        "## energy contrastと有効音声長の交差比較",
        "",
        "| contrast | 長さ | 受付 | 発話数 | FAR | FRR |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in results["cv_crossed"]:
        m = r["metrics"]
        lines.append(
            f"| {r['contrast']} | {r['duration']} | {r['policy']} | {m['queries']} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} |"
        )
    lines += [
        "",
        "## 検証と限界",
        "",
        f"独立float64再計算 {results['verification']['independent_float64_score_rows']:,}行、最大誤差 {results['verification']['max_score_error']:.3g}。12条件の加工再現・モデル再推論・元PCM特徴再抽出を確認。",
        "",
        "話者bootstrap 2,000回の区間・スコア分布・登録音声の品質・alignment変化は[JSON](evaluation-results.json)を参照。区間は固定モデル/閾値に条件付く探索的なもの。多数比較の補正はせず、自然音声の層別差を因果効果と解釈しない。合成劣化はCVの実録音を再現したものではない。新しい独立testや別日・別端末での性能保証は含まない。",
        "",
    ]
    return "\n".join(lines)


def main():
    config = s.read_json(t.CONFIG)
    run = t.ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    print("Auditing scores and waveform/model replay", flush=True)
    verification = audit(config, run)
    quality = list(s.rows(run / "quality.jsonl"))
    summary, comparisons = summarize_quality(quality, config)
    print("CV strata and speaker bootstrap", flush=True)
    strata, differences, crossed = quality_strata(config, run, quality)
    print("Stress metrics and paired bootstrap", flush=True)
    stress = stress_report(config, run)
    results = {
        "config": config,
        "quality_summary": summary,
        "quality_speaker_comparisons": comparisons,
        "cv_strata": strata,
        "cv_strata_differences": differences,
        "cv_crossed": crossed,
        "stress": stress,
        "verification": verification,
    }
    s.write_json(run / "results.json", results)
    s.write_json(t.BASE / "evaluation-results.json", results)
    (t.BASE / "evaluation-results.md").write_text(markdown(results))
    s.write_json(
        run / "completion-verification.json",
        {
            **verification,
            "files": {
                t.rel(p): s.sha256_file(p)
                for p in [
                    run / "results.json",
                    run / "quality.jsonl",
                    run / "quality-strata-freeze.json",
                    t.BASE / "evaluation-results.json",
                    t.BASE / "evaluation-results.md",
                    Path(__file__),
                ]
            },
        },
    )
    print("Reports complete", flush=True)


if __name__ == "__main__":
    main()
