"""Four fixed corpus/label-count conditions, followed by per-corpus evaluation."""

from __future__ import annotations

import argparse
import fcntl
import heapq
import json
import os
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict

from acquire import BASE, ROOT, rank, read, sha
from corpus import atomic, fixed

sys.path.insert(0, str(BASE.parent / "phoneme-greedy-selection"))
from common import (
    VOWELS,
    expanded,
    model_embeddings,
    np,
    pipeline,
    pool_features,
    rows,
    torch,
)
from phase3_train.metrics import (
    error_rates,
    far_target_threshold,
    json_threshold,
    roc_eer,
)
from scaling_learner import TrainingPool, load_model, train_trial

CONFIG = BASE / "study-config.json"
ARMS = ("baseline140", "mixed140", "expanded210", "expanded280")
CORPORA = ("jvs", "common_voice", "jtube")


def verify(run):
    frozen = read(run / "design-freeze.json")
    assert frozen["runtime"] == expanded.runtime()
    for name, digest in frozen["files"].items():
        assert sha(ROOT / name) == digest, name
    return frozen


def planned_speakers(existing, partition, cv_seed):
    jvs = sorted(s for s in existing if not s.startswith("cv17_"))
    cv = sorted(
        (s for s in existing if s.startswith("cv17_")),
        key=lambda s: (rank(cv_seed, s), s),
    )
    jtube = ["jtube_" + s.lower() for s in partition]
    assert len(jvs) == len(cv) == 70 and len(jtube) == 140
    result = {
        "baseline140": sorted(jvs + cv),
        "mixed140": sorted(jvs + cv[:35] + jtube[:35]),
        "expanded210": sorted(jvs + cv + jtube[:70]),
        "expanded280": sorted(jvs + cv + jtube),
    }
    assert set(result["expanded210"]) < set(result["expanded280"])
    return result


def freeze(config, run):
    source, corpus = ROOT / config["source_run"], ROOT / config["corpus_run"]
    assert (corpus / "preparation-freeze.json").exists()
    selection_path = (
        ROOT
        / read(corpus / "preparation-freeze.json")["config"]["source_run"]
        / "selection-freeze.json"
    )
    old = read(source / "design-freeze.json")
    assert old["runtime"] == expanded.runtime()
    for key in (
        "encoder",
        "training_seed",
        "maximum_updates",
        "warmup_updates",
        "feature_statistics_sha256",
    ):
        assert old["config"][key] == config[key], key
    paths = [
        CONFIG,
        BASE / "study.py",
        BASE / "scaling_learner.py",
        BASE / "audit_corpus.py",
        source / "design-freeze.json",
        source / "preparation-report.json",
        source / "train-segments.jsonl",
        source / "train-features.f32",
        corpus / "preparation-freeze.json",
        selection_path,
        ROOT / config["feature_statistics"],
    ]
    for filename in ("encoder.pt", "training-summary.json"):
        paths.append(source / "trials" / config["baseline_trial"] / filename)
    cv = ROOT / config["cv_evaluation_run"]
    paths.extend([cv / "design-freeze.json", cv / "completion-verification.json"])
    cv_pins = read(cv / "completion-verification.json")["output_sha256"]
    for split in ("validation", "test"):
        paths += [
            source / f"{split}-{suffix}"
            for suffix in ("features.f32", "segments.jsonl", "inputs.json")
        ]
        paths += [
            cv / split / name
            for name in (
                "features.npy",
                "segments.jsonl",
                "inputs.json",
                "utterances.jsonl",
                "preparation.json",
            )
        ]
    paths.extend(ROOT / name for name in old["files"] if name.endswith(".py"))
    paths.extend((BASE / "tests").glob("test_study.py"))
    # Validate reused baseline input bytes against its original freeze.
    for path in paths:
        name = str(path.relative_to(ROOT))
        if name in old["files"]:
            assert sha(path) == old["files"][name], name
    for path in paths:
        name = str(path.relative_to(ROOT))
        if name.startswith(config["cv_evaluation_run"] + "/") and name in cv_pins:
            assert sha(path) == cv_pins[name], name
    plans = planned_speakers(
        read(source / "preparation-report.json")["training_speakers"],
        read(selection_path)["partitions"]["train"],
        config["cv_subset_seed"],
    )
    summary = read(
        source / "trials" / config["baseline_trial"] / "training-summary.json"
    )
    assert summary["completed_updates"] == config["maximum_updates"]
    assert (
        sha(source / "trials" / config["baseline_trial"] / "encoder.pt")
        == summary["encoder_sha256"]
    )
    fixed(
        run / "design-freeze.json",
        {
            "config": config,
            "runtime": expanded.runtime(),
            "phonemes": old["phonemes"],
            "speaker_plans": plans,
            "files": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "previously_observed_test": {
                "jvs": True,
                "common_voice": True,
                "jtube": False,
            },
            "source_baseline_initial_encoder_sha256": summary["initial_encoder_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "stage": "comparison_frozen",
                "speaker_counts": {k: len(v) for k, v in plans.items()},
            }
        ),
        flush=True,
    )


def overlap_audit(config, run):
    corpus = ROOT / config["corpus_run"]
    known = set()
    for split in ("train", "validation", "test"):
        known.update(
            r["source_sha256"]
            for r in rows(ROOT / config["source_run"] / f"{split}-segments.jsonl")
        )
    for split in ("validation", "test"):
        known.update(
            r["source_sha256"]
            for r in rows(ROOT / config["cv_evaluation_run"] / split / "segments.jsonl")
        )
    identities = defaultdict(set)
    for split in ("train", "validation", "test"):
        for row in rows(corpus / f"{split}-segments.jsonl"):
            identities[row["source_sha256"]].add(row["original_mp3_sha256"])
    excluded = {
        digest: "matches_existing_model_or_evaluation_audio"
        if digest in known
        else "multiple_new_sources_have_identical_pcm_wav"
        for digest, sources in identities.items()
        if digest in known or len(sources) > 1
    }
    fixed(
        run / "corpus-overlap-audit.json",
        {
            "excluded_source_hashes": excluded,
            "policy": "exclude all matching new-source segments across every split; retain query slots as no score; no replacement",
        },
    )
    return set(excluded)


def verify_prepared(run):
    for name, digest in read(run / "prepared-inputs-freeze.json")["files"].items():
        assert sha(ROOT / name) == digest, name


def prepare_arms(config, run):
    frozen = verify(run)
    corpus, source = ROOT / config["corpus_run"], ROOT / config["source_run"]
    assert read(corpus / "pipeline-state.json")["status"] == "complete"
    assert read(corpus / "complete-audit.json")["receipts_verified"] == 9000
    prior = read(source / "preparation-report.json")
    new = read(corpus / "preparation-report.json")["splits"]["train"]
    sources = [(source, prior["train_feature_count"]), (corpus, new["segments"])]
    excluded = overlap_audit(config, run)
    input_pin = {
        str((p / f"train-{suffix}").relative_to(ROOT)): sha(p / f"train-{suffix}")
        for p, _ in sources
        for suffix in ("segments.jsonl", "features.f32")
    }
    for split in ("validation", "test"):
        for suffix in ("segments.jsonl", "features.f32"):
            path = corpus / f"{split}-{suffix}"
            input_pin[str(path.relative_to(ROOT))] = sha(path)
    path = run / "corpus-overlap-audit.json"
    input_pin[str(path.relative_to(ROOT))] = sha(path)
    fixed(
        run / "prepared-inputs-freeze.json",
        {
            "files": input_pin,
            "complete_audit_sha256": sha(corpus / "complete-audit.json"),
        },
    )
    for arm in ARMS:
        directory = run / arm
        directory.mkdir(exist_ok=True)
        if (directory / "design-freeze.json").exists():
            for name, digest in read(directory / "design-freeze.json")["files"].items():
                assert sha(ROOT / name) == digest
            continue
        allowed = set(frozen["speaker_plans"][arm])
        speakers, phone_counts, groups, corpora = set(), Counter(), Counter(), Counter()
        count = 0
        manifest, cache = (
            directory / "train-segments.jsonl",
            directory / "train-features.f32",
        )
        with manifest.open("w") as output, cache.open("wb") as features:
            for path, size in sources:
                if arm == "baseline140" and path == corpus:
                    continue
                matrix = np.memmap(
                    path / "train-features.f32",
                    dtype="<f4",
                    mode="r",
                    shape=(size, 128),
                )
                for index, row in enumerate(rows(path / "train-segments.jsonl")):
                    assert (
                        row["cache_index"] == index
                        and row["split"] == "train"
                        and not row["quality_flags"]
                    )
                    if path == corpus and row["source_sha256"] in excluded:
                        continue
                    if row["speaker_id"] not in allowed:
                        continue
                    output.write(
                        json.dumps({**row, "cache_index": count}, ensure_ascii=False)
                        + "\n"
                    )
                    features.write(matrix[index].tobytes())
                    count += 1
                    speakers.add(row["speaker_id"])
                    phone_counts[row["phoneme"]] += 1
                    groups[row["speaker_id"], row["phoneme"]] += 1
                    corpora[
                        "jtube"
                        if row["speaker_id"].startswith("jtube_")
                        else "common_voice"
                        if row["speaker_id"].startswith("cv17_")
                        else "jvs"
                    ] += 1
        assert speakers == allowed, (
            f"Unsupported planned speakers in {arm}: {allowed - speakers}"
        )
        assert all(
            any(p == phone and n >= 2 for (_, p), n in groups.items())
            for phone in frozen["phonemes"]
        ), f"No positive pair in {arm}"
        fixed(
            directory / "preparation-report.json",
            {
                "training_speakers": sorted(speakers),
                "train_feature_count": count,
                "phone_counts": dict(phone_counts),
                "corpus_segment_counts": dict(corpora),
                "speaker_phone_counts": {
                    s: {p: groups[s, p] for p in frozen["phonemes"]}
                    for s in sorted(speakers)
                },
            },
        )
        fixed(
            directory / "design-freeze.json",
            {
                "config": config,
                "arm": arm,
                "parent_design_sha256": sha(run / "design-freeze.json"),
                "files": {
                    str(p.relative_to(ROOT)): sha(p)
                    for p in (manifest, cache, directory / "preparation-report.json")
                },
            },
        )
        print(
            json.dumps(
                {
                    "stage": "arm_prepared",
                    "arm": arm,
                    "speakers": len(speakers),
                    "segments": count,
                }
            ),
            flush=True,
        )


def jtube_inputs(config, corpus, split, phones, excluded):
    original = [r for r in read(corpus / "inputs.json") if r["split"] == split]
    selected = list(rows(corpus / f"{split}-segments.jsonl"))
    by_source, enroll = defaultdict(lambda: {p: [] for p in phones}), defaultdict(list)
    for row in selected:
        if row["source_sha256"] in excluded:
            continue
        by_source[row["original_mp3_sha256"]][row["phoneme"]].append(row["cache_index"])
        if row["source_position"] < config["enrollment_clips"]:
            enroll[row["speaker_id"], row["phoneme"]].append(row)
    speakers = sorted({r["speaker_id"].lower() for r in original})
    profiles = {s: {} for s in speakers}
    for speaker in speakers:
        for phone in phones:
            pool = sorted(
                enroll[speaker, phone],
                key=lambda r: (
                    rank(config["enrollment_seed"], r["segment_id"]),
                    r["segment_id"],
                ),
            )
            profiles[speaker][phone] = [
                r["cache_index"] for r in pool[: config["enrollment_interval_limit"]]
            ]
    return {
        "split": split,
        "speakers": speakers,
        "profiles": profiles,
        "universally_registered": [
            p for p in phones if all(profiles[s][p] for s in speakers)
        ],
        "queries": [
            {
                "query_id": "jtube_" + r["sha256"][:24],
                "speaker_id": r["speaker_id"].lower(),
                "role": "verification",
                "groups": by_source[r["sha256"]],
            }
            for r in original
            if r["position"] >= config["enrollment_clips"]
        ],
    }


def score(data, vectors, phones):
    use = set(phones) & set(data["universally_registered"])
    profiles = {}
    for speaker in data["speakers"]:
        for phone in use:
            value = vectors[data["profiles"][speaker][phone]].mean(axis=0)
            norm = np.linalg.norm(value)
            assert norm > 1e-12
            profiles[speaker, phone] = value / norm
    result = []
    for query in data["queries"]:
        present = sorted(p for p in use if query["groups"].get(p))
        means = {p: vectors[query["groups"][p]].mean(axis=0) for p in present}
        for speaker in data["speakers"]:
            scorable = all(
                query["groups"].get(p) and data["profiles"][speaker].get(p) and p in use
                for p in VOWELS
            )
            result.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": speaker,
                    "role": query["role"],
                    "is_genuine": query["speaker_id"] == speaker,
                    "score": float(
                        np.mean(
                            [float(means[p] @ profiles[speaker, p]) for p in present]
                        )
                    )
                    if scorable
                    else None,
                }
            )
    return result


def operating(records, threshold):
    genuine = [r for r in records if r["is_genuine"]]
    impostor = [r for r in records if not r["is_genuine"]]
    scored = [r for r in records if r["score"] is not None]
    values = np.array([r["score"] for r in scored])
    labels = np.array([r["is_genuine"] for r in scored], dtype=bool)
    numeric = (
        np.inf if threshold == "+inf" else -np.inf if threshold == "-inf" else threshold
    )
    assert genuine and impostor
    fa = sum(r["score"] is not None and r["score"] >= numeric for r in impostor)
    fr = sum(r["score"] is None or r["score"] < numeric for r in genuine)
    rates = (
        error_rates(values, labels, numeric)
        if labels.any() and (~labels).any()
        else None
    )
    return {
        "all_genuine": len(genuine),
        "all_impostor": len(impostor),
        "scored_genuine": int(labels.sum()),
        "scored_impostor": int((~labels).sum()),
        "coverage": int(labels.sum()) / len(genuine),
        "eer": roc_eer(values, labels)["eer"] if rates else None,
        "far": fa / len(impostor),
        "frr": fr / len(genuine),
        "false_accepts": fa,
        "false_rejects": fr,
        "threshold": threshold,
        "scored_only_error_rates": rates,
    }


def evaluation_data(config, run, corpus_name, split, phones):
    if corpus_name == "jvs":
        path = ROOT / config["source_run"]
        data = read(path / f"{split}-inputs.json")
        features = np.memmap(
            path / f"{split}-features.f32",
            mode="r",
            dtype="<f4",
            shape=(data["feature_count"], 128),
        )
    elif corpus_name == "common_voice":
        path = ROOT / config["cv_evaluation_run"] / split
        data, features = (
            read(path / "inputs.json"),
            np.load(path / "features.npy", allow_pickle=False),
        )
    else:
        path = ROOT / config["corpus_run"]
        excluded = set(
            read(run / "corpus-overlap-audit.json")["excluded_source_hashes"]
        )
        data = jtube_inputs(config, path, split, phones, excluded)
        count = read(path / "preparation-report.json")["splits"][split]["segments"]
        features = np.memmap(
            path / f"{split}-features.f32", mode="r", dtype="<f4", shape=(count, 128)
        )
    trained = set(read(run / "design-freeze.json")["speaker_plans"]["expanded280"])
    assert not trained & set(data["speakers"])
    assert set(data["universally_registered"]) >= set(VOWELS), (
        f"Missing enrollment vowels for {corpus_name}/{split}; do not silently relax"
    )
    directory = run / "evaluation-inputs"
    fixed(directory / f"{corpus_name}-{split}.json", data)
    return data, features


def paired_intervals(records_by_arm, points, replicates, seed):
    speakers = sorted({r["speaker_id"] for r in records_by_arm["baseline140"]})
    index = {s: i for i, s in enumerate(speakers)}
    draws = np.random.default_rng(seed).multinomial(
        len(speakers), np.full(len(speakers), 1 / len(speakers)), size=replicates
    )
    rates = {}
    expected_keys = None
    for arm, records in records_by_arm.items():
        keys = [
            (r["query_id"], r["claimed_speaker_id"], r["is_genuine"]) for r in records
        ]
        if expected_keys is None:
            expected_keys = keys
        assert keys == expected_keys
        totals = np.zeros((len(speakers), 4), dtype=np.float64)
        threshold = (
            np.inf
            if points[arm] == "+inf"
            else -np.inf
            if points[arm] == "-inf"
            else points[arm]
        )
        for row in records:
            i = index[row["speaker_id"]]
            accepted = row["score"] is not None and row["score"] >= threshold
            if row["is_genuine"]:
                totals[i, 0] += not accepted
                totals[i, 1] += 1
            else:
                totals[i, 2] += accepted
                totals[i, 3] += 1
        sampled = draws @ totals
        rates[arm] = {
            "frr": sampled[:, 0] / sampled[:, 1],
            "far": sampled[:, 2] / sampled[:, 3],
        }
    return {
        arm: {
            metric: np.quantile(
                rates[arm][metric] - rates["baseline140"][metric], [0.025, 0.975]
            ).tolist()
            for metric in ("far", "frr")
        }
        for arm in ARMS
        if arm != "baseline140"
    }


def evaluate(config, run):
    frozen = verify(run)
    verify_prepared(run)
    phones = frozen["phonemes"]
    # Every final model exists before the first new validation score.
    models = {arm: load_model(run / arm / "trial")[0] for arm in ARMS}
    thresholds = {}
    for corpus_name in CORPORA:
        data, features = evaluation_data(config, run, corpus_name, "validation", phones)
        thresholds[corpus_name] = {}
        for arm, model in models.items():
            records = score(data, model_embeddings(model, features), phones)
            normal = [
                r
                for r in records
                if r["role"] == "verification" and r["score"] is not None
            ]
            values, labels = (
                np.array([r["score"] for r in normal]),
                np.array([r["is_genuine"] for r in normal]),
            )
            assert labels.any() and (~labels).any()
            value, _ = far_target_threshold(values, labels, 0.01)
            thresholds[corpus_name][arm] = json_threshold(value)
            fixed(run / "scores" / f"{corpus_name}-{arm}-validation.json", records)
    fixed(
        run / "evaluation-freeze.json",
        {
            "thresholds": thresholds,
            "models": {arm: sha(run / arm / "trial/encoder.pt") for arm in ARMS},
            "validation_inputs": {
                name: sha(run / "evaluation-inputs" / f"{name}-validation.json")
                for name in CORPORA
            },
            "test_used_for_selection": False,
        },
    )
    results, intervals = {}, {}
    for corpus_name in CORPORA:
        data, features = evaluation_data(config, run, corpus_name, "test", phones)
        results[corpus_name], paired = {}, {}
        for arm, model in models.items():
            records = score(data, model_embeddings(model, features), phones)
            fixed(run / "scores" / f"{corpus_name}-{arm}-test.json", records)
            paired[arm] = [r for r in records if r["role"] == "verification"]
            results[corpus_name][arm] = {
                role: operating(
                    [r for r in records if r["role"] == role],
                    thresholds[corpus_name][arm],
                )
                for role in sorted({r["role"] for r in records})
            }
        intervals[corpus_name] = paired_intervals(
            paired,
            thresholds[corpus_name],
            config["bootstrap_replicates"],
            config["bootstrap_seed"],
        )
    fixed(
        run / "results.json",
        {
            "conditions": results,
            "paired_speaker_bootstrap95_vs_baseline": intervals,
            "training_seeds": [config["training_seed"]],
            "interpretation": config["interpretation"],
            "evaluation_freeze_sha256": sha(run / "evaluation-freeze.json"),
        },
    )
    lines = [
        "# コーパス・話者数の拡張比較",
        "",
        "同じ初期encoder・216,000更新・固定最終checkpoint。1 seedの探索的比較。",
        "",
        "| コーパス | 条件 | EER（採点可能分） | FAR（全入力） | FRR（未採点含む） | 採点率 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, arms in results.items():
        for arm, roles in arms.items():
            r = roles["verification"]
            eer = "未算出" if r["eer"] is None else f"{r['eer'] * 100:.3f}%"
            lines.append(
                f"| {name} | {arm} | {eer} | {r['far'] * 100:.3f}% | {r['frr'] * 100:.3f}% | {r['coverage'] * 100:.2f}% |"
            )
    lines += [
        "",
        "JVS・Common Voiceのtestは過去にも観測済み。JTubeSpeechのテキストは自動生成。コーパス間の実在人物の重複は不明。",
        "話者数以外に音声量・録音条件・ラベル品質も変わるため、純粋な話者数の因果効果とはしない。",
    ]
    (run / "results.md").write_text("\n".join(lines) + "\n")


def check_feature_runtime(config, run):
    corpus = ROOT / config["corpus_run"]
    extraction = read(corpus / "extraction-freeze.json")
    pipe = pipeline(extraction["configuration"])
    report = {
        "reference_runtime": expanded.runtime(),
        "extractor_runtime_note": "torch 2.14.0; training reference torch 2.14.1",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        count = read(corpus / "preparation-report.json")["splits"][split]["segments"]
        cache = np.memmap(
            corpus / f"{split}-features.f32", dtype="<f4", mode="r", shape=(count, 128)
        )
        probes = heapq.nsmallest(
            config["feature_probes_per_split"],
            rows(corpus / f"{split}-segments.jsonl"),
            key=lambda r: rank(config["feature_probe_seed"], r["segment_id"]),
        )
        maximum = 0.0
        for row in probes:
            with wave.open(str(ROOT / row["source_file"])) as wav:
                wav.setpos(row["start_frame"])
                pcm = (
                    np.frombuffer(
                        wav.readframes(row["end_frame"] - row["start_frame"]),
                        dtype="<i2",
                    ).astype(np.float32)
                    / 32768
                )
            expected = pool_features(pipe, pcm, row["segment_id"])
            maximum = max(
                maximum, float(np.abs(expected - cache[row["cache_index"]]).max())
            )
        assert maximum <= config["feature_maximum_absolute_difference"], (
            "Feature runtime arithmetic differs"
        )
        report["splits"][split] = {
            "count": len(probes),
            "maximum_absolute_difference": maximum,
            "probe_ids": [r["segment_id"] for r in probes],
        }
    fixed(run / "feature-runtime-check.json", report)


def execute(config, run, wait):
    verify(run)
    state = run / "execution-state.json"
    corpus = ROOT / config["corpus_run"]
    while read(corpus / "pipeline-state.json")["status"] != "complete":
        current = read(corpus / "pipeline-state.json")
        if current["status"] == "failed" or not wait:
            raise RuntimeError(
                "Corpus preparation is not complete; training remains blocked"
            )
        atomic(
            state,
            {
                "status": "waiting_for_corpus",
                "pid": os.getpid(),
                "updated_at_epoch": time.time(),
            },
        )
        os.kill(current["pid"], 0)
        time.sleep(10)
    atomic(state, {"status": "auditing_corpus", "pid": os.getpid()})
    subprocess.run(
        [sys.executable, str(BASE / "audit_corpus.py")], cwd=ROOT, check=True
    )
    check_feature_runtime(config, run)
    prepare_arms(config, run)
    verify_prepared(run)
    frozen = verify(run)
    for arm in ARMS:
        atomic(state, {"status": "training", "arm": arm, "pid": os.getpid()})
        directory = run / arm
        trial = directory / "trial"
        if arm == "baseline140":
            trial.mkdir(exist_ok=True)
            source = ROOT / config["source_run"] / "trials" / config["baseline_trial"]
            for name in ("encoder.pt", "training-summary.json"):
                if not (trial / name).exists():
                    os.link(source / name, trial / name)
            load_model(trial)
        else:
            pool = TrainingPool(config, directory)
            train_trial(config, directory, trial, frozen["phonemes"], pool)
            del pool
        summary = read(trial / "training-summary.json")
        assert (
            summary["initial_encoder_sha256"]
            == frozen["source_baseline_initial_encoder_sha256"]
        )
        assert summary["completed_updates"] == config["maximum_updates"]
    atomic(state, {"status": "evaluating", "pid": os.getpid()})
    evaluate(config, run)
    atomic(
        state,
        {
            "status": "complete",
            "results": str((run / "results.md").relative_to(ROOT)),
            "finished_at_epoch": time.time(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "execute", "evaluate", "status"])
    parser.add_argument("--wait-for-corpus", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    if args.phase == "freeze":
        freeze(config, run)
    elif args.phase == "status":
        print(json.dumps(read(run / "execution-state.json"), ensure_ascii=False))
    elif args.phase == "evaluate":
        evaluate(config, run)
    else:
        with (run / "execution.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                execute(config, run, args.wait_for_corpus)
            except BaseException as error:
                atomic(
                    run / "execution-state.json",
                    {
                        "status": "failed",
                        "error": str(error),
                        "updated_at_epoch": time.time(),
                    },
                )
                raise


if __name__ == "__main__":
    main()
