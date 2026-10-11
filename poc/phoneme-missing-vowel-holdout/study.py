"""A single preregistered, speaker-disjoint evaluation of relaxed vowel support."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-all-missing-vowel-evaluation"))
import shared as s


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


primitives = module(
    "holdout_primitives", BASE.parent / "phoneme-greedy-selection/prepare.py"
)
preparation = module(
    "holdout_preparation",
    BASE.parent / "phoneme-training-data-expansion/prepare_data.py",
)
bootstrap = module(
    "holdout_bootstrap",
    BASE.parent / "phoneme-all-missing-vowel-evaluation/auditing.py",
)
from phase3_train.metrics import error_rates, roc_eer

CONFIG = BASE / "config.json"


def digest(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def verify(config, run):
    frozen = s.read_json(run / "design-freeze.json")
    if frozen["config"] != config:
        raise ValueError("protocol changed after freeze")
    for path, checksum in frozen["files"].items():
        s.checked(ROOT / path, checksum)
    return frozen


def freeze(config, run):
    from pyarrow import parquet

    if run.exists():
        raise ValueError("unused run required")
    source = ROOT / config["cv_source"]
    selection = s.read_json(source / "selection.json")
    trained_clients = {r["client_id"] for r in selection}
    trained_speakers = {r["speaker_id"] for r in selection}
    if len(trained_clients) != 70:
        raise ValueError("training client list changed")
    files = {}
    paths = [
        CONFIG,
        BASE / "README.md",
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        source / "selection.json",
        source / "source/manifest.json",
        source / "additional-utterances.jsonl",
    ]
    paths.extend(
        [
            Path(s.__file__),
            Path(bootstrap.__file__),
            Path(primitives.__file__),
            Path(preparation.__file__),
        ]
    )
    source_manifest = s.read_json(source / "source/manifest.json")
    by_client = defaultdict(list)
    for shard, checksum in source_manifest["files"].items():
        path = source / "source" / shard
        s.checked(path, checksum)
        paths.append(path)
        if not shard.endswith(".parquet"):
            continue
        for index, row in enumerate(
            parquet.read_table(
                path, columns=["client_id", "sentence", "locale"]
            ).to_pylist()
        ):
            if (
                row["client_id"] not in trained_clients
                and row["client_id"]
                and row["sentence"]
                and row["locale"] == "ja"
            ):
                by_client[row["client_id"]].append(
                    {**row, "shard": shard, "shard_row": index}
                )
    clients = sorted(
        [p for p, rows in by_client.items() if len(rows) >= config["minimum_clips"]],
        key=lambda p: digest(config["speaker_seed"], p),
    )
    count = config["calibration_speakers"] + config["test_speakers"]
    if len(clients) < count:
        raise ValueError("insufficient unused speakers")
    selected = []
    for number, client in enumerate(clients[:count]):
        split = "validation" if number < config["calibration_speakers"] else "test"
        speaker = "cv17_" + preparation.stable_hash("speaker_id", client)[:16]
        if speaker in trained_speakers:
            raise ValueError("training speaker overlap")
        clips = sorted(
            by_client[client],
            key=lambda r: digest(
                config["clip_seed"], f"{r['shard']}/{r['shard_row']}/{r['sentence']}"
            ),
        )
        for index, row in enumerate(
            clips[: config["enrollment_clips"] + config["query_clips"]]
        ):
            selected.append(
                {
                    **row,
                    "speaker_id": speaker,
                    "split": split,
                    "role": "enrollment"
                    if index < config["enrollment_clips"]
                    else "verification",
                }
            )
    encoder = (
        ROOT / config["encoder_run"] / "trials" / config["encoder_trial"] / "encoder.pt"
    )
    s.checked(encoder, config["encoder_sha256"])
    model, bundle = s.original.load_model(encoder.parent)
    del model
    source_design = s.read_json(ROOT / config["encoder_run"] / "design-freeze.json")
    extract_config = {
        **source_design["config"],
        "minimum_frames": 720,
        "maximum_frames": 6000,
        "minimum_rms_dbfs": -50,
    }
    paths.extend(
        [
            encoder,
            encoder.parent / "training-summary.json",
            ROOT / config["encoder_run"] / "design-freeze.json",
            ROOT / extract_config["feature_statistics"],
        ]
    )
    # Preserve the complete original source provenance; verification includes old audio hashes.
    files.update(source_design["files"])
    for path in paths:
        s.original.pin(files, path)
    for loaded in tuple(sys.modules.values()):
        filename = getattr(loaded, "__file__", None)
        if (
            filename
            and filename.endswith(".py")
            and Path(filename).resolve().is_relative_to(ROOT / "poc")
        ):
            s.original.pin(files, Path(filename))
    jvs_points = {}
    prior = s.read_json(
        ROOT / "poc/phoneme-all-missing-vowel-evaluation/config/protocol.json"
    )
    path = ROOT / prior["run_directory"] / "threshold-freeze.json"
    s.original.pin(files, path)
    jvs_points = s.read_json(path)["thresholds"]["equal"]
    # Reserve and hash-check all selected compressed audio without scoring it.
    needed = {(r["shard"], r["shard_row"]): r for r in selected}
    hashes = defaultdict(list)
    for shard in sorted({r["shard"] for r in selected}):
        for index, audio in enumerate(
            parquet.read_table(source / "source" / shard, columns=["audio"]).to_pylist()
        ):
            key = (shard, index)
            if key not in needed:
                continue
            row = needed[key]
            row["audio_path"] = audio["audio"]["path"]
            row["original_audio_sha256"] = hashlib.sha256(
                audio["audio"]["bytes"]
            ).hexdigest()
            hashes[row["original_audio_sha256"]].append(key)
    old_raw = {
        r["original_audio_sha256"]
        for r in s.rows(source / "additional-utterances.jsonl")
    }
    duplicates = {h for h, keys in hashes.items() if len(keys) > 1 or h in old_raw}
    for row in selected:
        row["excluded_duplicate"] = row["original_audio_sha256"] in duplicates
    run.mkdir(parents=True)
    s.write_json(run / "selection.json", selected)
    s.original.pin(files, run / "selection.json")
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "phones": bundle["phonemes"],
            "extract_config": extract_config,
            "jvs_thresholds": jvs_points,
            "eligible_unused_clients": len(clients),
            "selected_speakers": count,
            "duplicate_excluded_clips": sum(r["excluded_duplicate"] for r in selected),
            "status": "selection_protocol_and_model_fixed_before_new_audio_processing_or_scoring",
        },
    )
    print(
        f"Frozen {count} new speakers, {len(selected)} clips, {len(duplicates)} duplicate hashes",
        flush=True,
    )


def prepare_split(config, run, split, design):
    import pyopenjtalk
    from pyarrow import parquet

    destination = run / split
    if destination.exists():
        raise ValueError("split already exists; use a fresh run")
    destination.mkdir()
    selected = [
        dict(r) for r in s.read_json(run / "selection.json") if r["split"] == split
    ]
    needed = {(r["shard"], r["shard_row"]): r for r in selected}
    for shard in sorted({r["shard"] for r in selected}):
        path = ROOT / config["cv_source"] / "source" / shard
        for i, audio in enumerate(
            parquet.read_table(path, columns=["audio"]).to_pylist()
        ):
            if (shard, i) in needed:
                row = needed[shard, i]
                row["audio_bytes"] = audio["audio"]["bytes"]
                if (
                    hashlib.sha256(row["audio_bytes"]).hexdigest()
                    != row["original_audio_sha256"]
                ):
                    raise ValueError("compressed audio changed")
    dictionary = pyopenjtalk.OPEN_JTALK_DICT_DIR
    if isinstance(dictionary, bytes):
        dictionary = dictionary.decode()
    version = importlib.metadata.version("pyopenjtalk")

    def process(row):
        value = preparation.prepare_clip(row, destination, version, dictionary)
        # The upstream helper is for training; override metadata in this new evaluation run.
        meta = value["utterance"]
        meta.update(split=split, evaluation_role=row["role"], learning_curve_cohorts=[])
        s.write_json(destination / "prepared" / f"{meta['utterance_id']}.json", value)
        return row, value

    prepared = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(process, row) for row in selected]
        for i, future in enumerate(as_completed(pending), 1):
            prepared.append(future.result())
            if i % 100 == 0 or i == len(pending):
                print(f"{split} alignment {i}/{len(pending)}", flush=True)
    prepared.sort(key=lambda pair: (pair[0]["shard"], pair[0]["shard_row"]))
    # Check decoded PCM as well as compressed bytes for old-source and split overlap.
    old_hashes = {
        r["source_sha256"]
        for which in ("train", "validation", "test")
        for r in s.rows(ROOT / config["encoder_run"] / f"{which}-segments.jsonl")
    }
    if split == "test":
        old_hashes.update(
            r["source_sha256"] for r in s.rows(run / "validation/utterances.jsonl")
        )
    pcm_counts = Counter(v["utterance"]["source_sha256"] for _, v in prepared)
    pipe = primitives.pipeline(design["extract_config"])
    segments, features, utterances, excluded = [], [], [], []
    groups_by_source = {}
    files, qc = {}, Counter()
    for row, value in prepared:
        meta = value["utterance"]
        meta["role"] = row["role"]
        meta["excluded_duplicate"] = (
            row["excluded_duplicate"]
            or pcm_counts[meta["source_sha256"]] > 1
            or meta["source_sha256"] in old_hashes
        )
        utterances.append(meta)
        groups = {p: [] for p in design["phones"]}
        groups_by_source[meta["source_file"]] = groups
        alignment = (
            destination
            / "alignments/raw"
            / meta["speaker_id"]
            / f"{meta['utterance_id']}.json"
        )
        if meta["excluded_duplicate"] or not alignment.exists():
            excluded.append(
                {
                    "utterance_id": meta["utterance_id"],
                    "reason": "duplicate_audio"
                    if meta["excluded_duplicate"]
                    else "alignment_failed",
                }
            )
            continue
        for segment, clip in primitives.extract(
            alignment, meta, design["extract_config"], files, qc
        ):
            if segment["quality_flags"] or segment["phoneme"] not in groups:
                continue
            segment["cache_index"] = len(segments)
            groups[segment["phoneme"]].append(len(segments))
            features.append(primitives.pool_features(pipe, clip, segment["segment_id"]))
            segments.append(segment)
    s.write_rows(destination / "utterances.jsonl", utterances)
    s.write_rows(destination / "segments.jsonl", segments)
    s.np.save(
        destination / "features.npy",
        s.np.asarray(features, dtype=s.np.float32),
        allow_pickle=False,
    )
    speakers = sorted({r["speaker_id"] for r in selected})
    profiles = {}
    for speaker in speakers:
        candidates = [
            r
            for r in segments
            if r["speaker_id"] == speaker and r["role"] == "enrollment"
        ]
        profiles[speaker] = {
            p: [
                r["cache_index"]
                for r in primitives.diverse(
                    [r for r in candidates if r["phoneme"] == p],
                    config["enrollment_limit"],
                    config["clip_seed"],
                )
            ]
            for p in design["phones"]
        }
    universal = [
        p for p in design["phones"] if all(profiles[speaker][p] for speaker in speakers)
    ]
    queries = [
        {
            "query_id": u["utterance_id"],
            "speaker_id": u["speaker_id"],
            "source_file": u["source_file"],
            "role": "verification",
            "groups": groups_by_source[u["source_file"]],
        }
        for u in utterances
        if u["role"] == "verification"
    ]
    data = {
        "split": split,
        "speakers": speakers,
        "profiles": profiles,
        "universally_registered": universal,
        "queries": queries,
    }
    s.write_json(destination / "inputs.json", data)
    s.write_json(
        destination / "preparation.json",
        {
            "files": files,
            "qc": dict(qc),
            "excluded": excluded,
            "utterances": len(utterances),
            "queries": len(queries),
            "intervals": len(segments),
            "universal_phones": universal,
            "missing_enrollment_vowels": {
                speaker: [p for p in s.VOWELS if not profiles[speaker][p]]
                for speaker in speakers
            },
            "recording_seconds": sum(u["duration_sec"] for u in utterances),
            "enrollment_seconds": sum(
                u["duration_sec"] for u in utterances if u["role"] == "enrollment"
            ),
        },
    )
    model, _ = s.original.load_model(
        ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
    )
    embeddings = s.original.model_embeddings(
        model, s.np.asarray(features, dtype=s.np.float32)
    )
    s.np.save(destination / "embeddings.npy", embeddings, allow_pickle=False)
    return data, embeddings


def candidates(data, vectors):
    use = data["universally_registered"]
    profiles = {}
    for speaker in data["speakers"]:
        for p in use:
            mean = vectors[data["profiles"][speaker][p]].mean(axis=0)
            profiles[speaker, p] = mean / s.np.linalg.norm(mean)
    result = []
    for query in data["queries"]:
        available = sorted(p for p in use if query["groups"][p])
        means = {p: vectors[query["groups"][p]].mean(axis=0) for p in available}
        for speaker in data["speakers"]:
            components = {p: float(means[p] @ profiles[speaker, p]) for p in available}
            result.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": speaker,
                    "is_genuine": speaker == query["speaker_id"],
                    "role": "verification",
                    "split": data["split"],
                    "status": "scored" if available else "no_score",
                    "score": float(s.np.mean(list(components.values())))
                    if available
                    else None,
                    "phone_scores": components,
                    "used_phones": available,
                }
            )
    return result


def metrics(values, points):
    scored = [r for r in values if r["status"] == "scored"]
    scores = s.np.array([r["score"] for r in scored])
    genuine = s.np.array([r["is_genuine"] for r in scored])
    total_g = sum(r["is_genuine"] for r in values)
    total_i = len(values) - total_g
    result = {
        "all_queries": total_g,
        "scored_queries": int(genuine.sum()),
        "coverage": float(genuine.sum() / total_g),
        "eer": roc_eer(scores, genuine)["eer"],
        "operating_points": {},
    }
    for name, threshold in points.items():
        rates = error_rates(scores, genuine, float(threshold))
        result["operating_points"][name] = {
            **rates,
            "threshold": threshold,
            "all_input_far": rates["false_accepts"] / total_i,
            "all_input_frr": (rates["false_rejects"] + total_g - int(genuine.sum()))
            / total_g,
        }
    return result


def evaluate(config, run, split):
    design = verify(config, run)
    if split == "test":
        freeze = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", freeze["design_sha256"])
        for path, checksum in freeze["files"].items():
            s.checked(ROOT / path, checksum)
    data, vectors = prepare_split(config, run, split, design)
    values = candidates(data, vectors)
    s.write_rows(run / split / "candidates.jsonl", values)
    reports, all_points = {}, {}
    for policy in config["policies"]:
        name = policy["name"]
        chosen = s.apply_policy(values, policy)
        points = (
            s.original.thresholds(chosen)
            if split == "validation"
            else freeze["thresholds"][name]
        )
        all_points[name] = points
        reports[name] = {
            "thresholds": points,
            "calibrated": metrics(chosen, points),
            "jvs_transfer": metrics(chosen, design["jvs_thresholds"][name]),
        }
        print(
            json.dumps(
                {"split": split, "policy": name, "result": reports[name]["calibrated"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    s.write_json(run / split / "results.json", reports)
    verify(config, run)
    if split == "validation":
        files = {
            str(p.relative_to(ROOT)): s.sha256_file(p)
            for p in (run / split).glob("*")
            if p.is_file()
        }
        s.write_json(
            run / "threshold-freeze.json",
            {
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "thresholds": all_points,
                "files": files,
                "status": "all_thresholds_frozen_before_test_audio_processing",
            },
        )


def report(config, run):
    design = verify(config, run)
    frozen = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", frozen["design_sha256"])
    for path, checksum in frozen["files"].items():
        s.checked(ROOT / path, checksum)
    results = {
        split: s.read_json(run / split / "results.json")
        for split in ("validation", "test")
    }
    score_checks = rate_checks = 0
    max_error = 0.0
    for split in results:
        data = s.read_json(run / split / "inputs.json")
        report_data = s.read_json(run / split / "preparation.json")
        for path, checksum in report_data["files"].items():
            s.checked(ROOT / path, checksum)
        vectors = s.np.load(run / split / "embeddings.npy", allow_pickle=False).astype(
            s.np.float64
        )
        values = list(s.rows(run / split / "candidates.jsonl"))
        references = candidates(data, vectors)
        for row, ref in zip(values, references, strict=True):
            for key in (
                "query_id",
                "speaker_id",
                "claimed_speaker_id",
                "is_genuine",
                "split",
                "status",
                "used_phones",
            ):
                if row[key] != ref[key]:
                    raise ValueError("score identity mismatch")
            if row["score"] is not None:
                max_error = max(
                    max_error,
                    abs(row["score"] - ref["score"]),
                    *[
                        abs(row["phone_scores"][p] - ref["phone_scores"][p])
                        for p in row["used_phones"]
                    ],
                )
            score_checks += 1
        if max_error > 2e-6:
            raise ValueError("float64 score reconstruction differs")
        # Check cached pooling and model output against original PCM for each available phone.
        model, _ = s.original.load_model(
            ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
        )
        features = s.np.load(run / split / "features.npy", allow_pickle=False)
        recalculated = s.original.model_embeddings(model, features)
        if not s.np.array_equal(recalculated, vectors.astype(s.np.float32)):
            raise ValueError("encoder replay differs")
        import wave

        seen = set()
        pipe = primitives.pipeline(design["extract_config"])
        for row in s.rows(run / split / "segments.jsonl"):
            if row["phoneme"] in seen:
                continue
            seen.add(row["phoneme"])
            with wave.open(str(ROOT / row["source_file"])) as wav:
                wav.setpos(row["start_frame"])
                signal = (
                    s.np.frombuffer(
                        wav.readframes(row["end_frame"] - row["start_frame"]),
                        dtype="<i2",
                    ).astype(s.np.float32)
                    / 32768
                )
            actual = primitives.pool_features(pipe, signal, row["segment_id"])
            if not s.np.array_equal(actual, features[row["cache_index"]]):
                raise ValueError("waveform feature replay differs")
        for policy in config["policies"]:
            name = policy["name"]
            chosen = s.apply_policy(values, policy)
            scored = [r for r in chosen if r["status"] == "scored"]
            scores = s.np.array([r["score"] for r in scored])
            genuine = s.np.array([r["is_genuine"] for r in scored])
            for mode in ("calibrated", "jvs_transfer"):
                metric = results[split][name][mode]
                if (
                    abs(s.original.independent_eer(scores, genuine) - metric["eer"])
                    > 1e-12
                ):
                    raise ValueError("independent EER differs")
                total_g = sum(r["is_genuine"] for r in chosen)
                for point in metric["operating_points"].values():
                    threshold = float(point["threshold"])
                    fa = sum(
                        r["score"] >= threshold and not r["is_genuine"] for r in scored
                    )
                    fr = sum(r["score"] < threshold and r["is_genuine"] for r in scored)
                    if fa != point["false_accepts"] or fr != point["false_rejects"]:
                        raise ValueError("independent error counts differ")
                    if (
                        abs(point["all_input_far"] - fa / (len(chosen) - total_g))
                        > 1e-12
                        or abs(
                            point["all_input_frr"]
                            - (fr + total_g - genuine.sum()) / total_g
                        )
                        > 1e-12
                    ):
                        raise ValueError("independent full-input rates differ")
                    rate_checks += 4
        print(f"Independent audit passed: {split}", flush=True)
    speakers = s.read_json(run / "test/inputs.json")["speakers"]
    lookup = {speaker: i for i, speaker in enumerate(speakers)}
    indices, counts = s.original.speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    values = list(s.rows(run / "test/candidates.jsonl"))
    draws, intervals, differences = {}, {}, {}
    bootstrap_checks = 0
    patterns = {}
    for policy in config["policies"]:
        name = policy["name"]
        chosen = s.apply_policy(values, policy)
        patterns[name] = {}
        for mode in ("calibrated", "jvs_transfer"):
            metric = results["test"][name][mode]
            points = {k: v["threshold"] for k, v in metric["operating_points"].items()}
            estimates, checks = bootstrap.paired_draws(chosen, points, counts, lookup)
            bootstrap_checks += checks
            for key, estimates_for_key in estimates.items():
                path = f"{name}/{mode}/{key}"
                draws[path] = estimates_for_key
                intervals[path] = s.np.percentile(
                    estimates_for_key, [2.5, 97.5]
                ).tolist()
                if name != "vowels5":
                    before = results["test"]["vowels5"][mode]
                    differences[path] = {
                        "difference": bootstrap.estimate(metric, key)
                        - bootstrap.estimate(before, key),
                        "ci95": s.np.percentile(
                            estimates_for_key - draws[f"vowels5/{mode}/{key}"],
                            [2.5, 97.5],
                        ).tolist(),
                    }
            groups = defaultdict(list)
            for row in chosen:
                missing = (
                    "".join(p for p in s.VOWELS if p not in row["used_phones"])
                    or "none"
                )
                groups[missing].append(row)
            for missing, group in groups.items():
                g = [r for r in group if r["is_genuine"]]
                i = [r for r in group if not r["is_genuine"]]
                t = float(points["far_1pct"])
                patterns[name][f"{mode}/{missing}"] = {
                    "queries": len(g),
                    "speakers": len({r["speaker_id"] for r in g}),
                    "scored_queries": sum(r["status"] == "scored" for r in g),
                    "false_accepts": sum(
                        r["status"] == "scored" and r["score"] >= t for r in i
                    ),
                    "impostor_trials": len(i),
                    "all_false_rejects": sum(
                        r["status"] != "scored" or r["score"] < t for r in g
                    ),
                }
    gate = config["research_gate"]
    criteria = {
        "far_upper95_at_most_1pct": intervals["vowels4/calibrated/far_1pct/far"][1]
        <= gate["test_far_upper95_max"],
        "far_increase_upper95_at_most_0_1pp": differences[
            "vowels4/calibrated/far_1pct/far"
        ]["ci95"][1]
        <= gate["far_difference_upper95_max"],
        "frr_improvement_upper95_below_zero": differences[
            "vowels4/calibrated/far_1pct/frr"
        ]["ci95"][1]
        < gate["frr_difference_upper95_max"],
    }
    decision = (
        "research_prototype_candidate_passed"
        if all(criteria.values())
        else "adoption_deferred"
    )
    audit = {
        "score_checks": score_checks,
        "rate_checks": rate_checks,
        "bootstrap_checks": bootstrap_checks,
        "float64_max_error": max_error,
        "status": "passed",
    }
    output = {
        "config": config,
        "results": results,
        "intervals95": intervals,
        "differences": differences,
        "patterns": patterns,
        "decision": decision,
        "criteria": criteria,
        "audit": audit,
        "preparation": {
            split: {
                k: v
                for k, v in s.read_json(run / split / "preparation.json").items()
                if k != "files"
            }
            for split in results
        },
    }
    s.np.savez_compressed(
        run / "bootstrap.npz", speaker_indices=indices, speaker_counts=counts, **draws
    )
    s.write_json(run / "results.json", output)
    s.write_json(BASE / "evaluation-results.json", output)
    lines = [
        "# 新規Common Voice話者での母音不足対応",
        "",
        f"判定: **{decision}**。事前固定した研究用基準の結果は {criteria}。",
        "",
        "学習に使っていない60 client_idを校正30・test30に固定。各話者20発話で登録し、30発話を照合。全36音素encoder・等重み・登録上限30区間を固定。testを見て閾値や条件を選び直していない。",
        "",
        "## test全入力の性能",
        "",
        "| 閾値 | 条件 | 採点可能 | EER | FAR | FRR | 他人誤受入 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in ("calibrated", "jvs_transfer"):
        for policy in config["policies"]:
            name = policy["name"]
            m = results["test"][name][mode]
            o = m["operating_points"]["far_1pct"]
            lines.append(
                f"| {mode} | {name} | {m['scored_queries']}/{m['all_queries']} | {100 * m['eer']:.3f}% | {100 * o['all_input_far']:.3f}% | {100 * o['all_input_frr']:.3f}% | {o['false_accepts']} |"
            )
    lines.extend(
        [
            "",
            "## 主比較の信頼区間",
            "",
            "| 指標 | 差または値 | 95%区間 |",
            "| --- | --- | --- |",
        ]
    )
    for key in ("far_1pct/far", "far_1pct/frr", "eer"):
        v = differences[f"vowels4/calibrated/{key}"]
        lines.append(
            f"| 4母音−5母音 {key} | {100 * v['difference']:+.3f} pp | [{100 * v['ci95'][0]:+.3f}, {100 * v['ci95'][1]:+.3f}] pp |"
        )
    lo, hi = intervals["vowels4/calibrated/far_1pct/far"]
    lines.append(
        f"| 4母音FAR | {100 * results['test']['vowels4']['calibrated']['operating_points']['far_1pct']['all_input_far']:.3f}% | [{100 * lo:.3f}, {100 * hi:.3f}]% |"
    )
    lines.extend(
        [
            "",
            "## 欠損別の内訳（新校正閾値・目標FAR 1%）",
            "",
            "| 条件 | 欠損母音 | 話者数 | 本人入力 | 採点可能 | 本人拒否 | 他人誤受入/試行 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for policy in config["policies"]:
        name = policy["name"]
        for key, p in sorted(patterns[name].items()):
            if key.startswith("calibrated/"):
                lines.append(
                    f"| {name} | {key.split('/')[1]} | {p['speakers']} | {p['queries']} | {p['scored_queries']} | {p['all_false_rejects']} | {p['false_accepts']}/{p['impostor_trials']} |"
                )
    lines.extend(
        [
            "",
            "## 限界と監査",
            "",
            "30 test話者を共有する2,000回の対応付きbootstrap。固定モデル・閾値に条件付いた区間で、校正の不確実性や多重比較補正を含まない。元コーパスは学習と同じCommon Voiceだがclient_idは分離。実在人物の重複と録音sessionは不明で、別日・別端末の保証ではない。",
            "",
            f"監査: {audit}",
            "",
            "[固定設計と再実行](README.md)・[全指標JSON](evaluation-results.json)・[結論](interpretation.md)",
        ]
    )
    (BASE / "evaluation-results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    verify(config, run)
    s.write_json(
        run / "completion-verification.json",
        {
            "status": "passed",
            "decision": decision,
            "output_sha256": {
                str(p.relative_to(ROOT)): s.sha256_file(p)
                for p in run.rglob("*")
                if p.is_file() and p.name != "completion-verification.json"
            },
        },
    )
    print(
        json.dumps({"decision": decision, "criteria": criteria, "audit": audit}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "calibrate", "test", "report"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    settings = s.read_json(CONFIG)
    run = args.run or ROOT / settings["run_directory"]
    if args.stage == "freeze":
        freeze(settings, run)
    elif args.stage == "report":
        report(settings, run)
    else:
        evaluate(settings, run, "validation" if args.stage == "calibrate" else "test")
