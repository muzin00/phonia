"""Frozen-encoder enrollment scaling; existing JVS test is exploratory."""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
for project in (
    "phoneme-speaker-encoder",
    "phoneme-speaker-encoder/scripts",
    "phoneme-user-registration",
    "phoneme-verification",
    "phoneme-verification-evaluation",
    "phoneme-baseline-comparison",
):
    sys.path.insert(0, str(ROOT / "poc" / project))

import numpy as np
import torch
from comparison_inputs import describe_audio, effective_window
from evaluation.inference import model_snapshot, require_unchanged
from evaluation.pipeline import load_protocol
from evaluation.trials import build_trials, prepare_split, to_input
from final_metrics import bootstrap_cell, interval
from frozen_test import BoundedAudio as TestAudio
from phase3_data.artifacts import make_enrollment
from phase3_data.manifest import VOWELS, iter_segments
from pilot import EmbeddingCache, read_rows, vowel_scores
from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    load_profile,
    register_user,
    save_profile,
)
from smoke import sha256_file, write_json
from validation import BoundedAudio as ValidationAudio
from validation import iter_rows, writer
from validation_metrics import POINTS, calibrate_cell, evaluate_cell, numeric_threshold
from verification import IncompleteVerification, VerificationInput, verify
from verification.inputs import prepare_query
from weighted_bootstrap import speaker_draws


def checked(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"frozen file changed: {path}")


def expanded_enrollment(segments, split, speakers, counts, seed):
    if counts != sorted(set(counts)) or not counts or counts[0] < 1:
        raise ValueError("positive increasing enrollment counts required")
    selected = make_enrollment(segments, split=split, seed=seed, count=max(counts))
    lookup = {s.segment_id: s for s in segments}
    if len(lookup) != len(segments) or len(set(speakers)) != len(speakers):
        raise ValueError("duplicate enrollment segment or speaker")
    if {r["speaker_id"] for r in selected} != set(speakers):
        raise ValueError("enrollment speaker mismatch")
    result = []
    for speaker in speakers:
        for count in counts:
            ranks = [
                r for r in selected if r["speaker_id"] == speaker and r["rank"] <= count
            ]
            if Counter(r["vowel"] for r in ranks) != Counter(
                {v: count for v in VOWELS}
            ):
                raise ValueError("incomplete enrollment")
            result.append(
                {
                    "schema_version": 1,
                    "split": split,
                    "user_id": speaker,
                    "enrollment_count": count,
                    "ranks": ranks,
                    "segments": [to_input(lookup[r["segment_id"]]) for r in ranks],
                }
            )
    return result


def prepare(run):
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("requires unused run directory inside artifacts")
    config_path = BASE / "config/protocol.json"
    config = json.loads(config_path.read_text())
    if (
        config["counts_per_vowel"] != [10, 20, 30]
        or config["method"] != "vowel_exact"
        or config["query_window"] != "full"
        or config["retrain_encoder"]
        or config["change_feature_statistics"]
        or config["threshold_recalibration_on_test"]
        or config["required_vowels"] != list(VOWELS)
        or config["roles"] != ["verification", "cross_text_verification"]
        or config["enrollment_seed"] != 20260930
        or config["repeats_per_unique_embedding"] != 3
        or config["bootstrap_replicates"] != 10000
        or config["bootstrap_seed"] != 20260929
        or config["operating_points"] != [p for p, _ in POINTS]
        or config["calibration"] != "validation_verification_only_per_enrollment_count"
        or config["acceptance"] != "score_gte_threshold"
        or config["no_score"] != "reject_and_keep_in_all_input_denominators"
    ):
        raise ValueError("unsupported scaling protocol")
    phase6 = ROOT / config["phase6_run"]
    checked(ROOT / config["phase6_protocol"], config["phase6_protocol_sha256"])
    checked(phase6 / "execution.json", config["phase6_execution_sha256"])
    old = load_protocol(ROOT / config["phase6_protocol"])
    execution = json.loads((phase6 / "execution.json").read_text())
    if execution["status"] != "completed":
        raise ValueError("Phase 6 incomplete")
    files = {str(config_path.relative_to(ROOT)): sha256_file(config_path)}
    files[config["phase6_protocol"]] = config["phase6_protocol_sha256"]
    files[str((phase6 / "execution.json").relative_to(ROOT))] = config[
        "phase6_execution_sha256"
    ]
    for row in [*old["inputs"].values(), *old["reused_files"]]:
        files[row["path"]] = row["sha256"]
    for name, checksum in old["model"]["bundle_sha256"].items():
        files[str(Path(old["model"]["bundle"]) / name)] = checksum
    for split in ("validation", "test"):
        prefix = f"phase7_{split}"
        source = ROOT / config[f"{prefix}_run"]
        report_path = source / f"{split}-report.json"
        checked(report_path, config[f"{prefix}_report_sha256"])
        report = json.loads(report_path.read_text())
        if report["status"] != "completed":
            raise ValueError("Phase 7 incomplete")
        files[str(report_path.relative_to(ROOT))] = sha256_file(report_path)
        for name in (
            ("validation-thresholds.json", "validation-metrics.json")
            if split == "validation"
            else ("test-metrics.json", "bootstrap-counts.npy")
        ):
            checked(source / name, report["outputs_sha256"][name])
            files[str((source / name).relative_to(ROOT))] = report["outputs_sha256"][
                name
            ]
        for name in (
            f"inputs/{split}/queries.jsonl",
            f"inputs/{split}/enrollment.jsonl",
            f"scores/{split}.jsonl",
            "thresholds/validation.json",
        ):
            checked(phase6 / name, execution["files"][name])
            files[str((phase6 / name).relative_to(ROOT))] = execution["files"][name]
    enrollment_segments = defaultdict(list)
    for segment in iter_segments(ROOT / old["inputs"]["segment_manifest"]["path"]):
        if segment.split in ("validation", "test") and segment.role == "enrollment":
            enrollment_segments[segment.split].append(segment)
    original_sources = read_rows(ROOT / old["inputs"]["utterance_manifest"]["path"])
    prepared = {}
    for split in ("validation", "test"):
        inputs = prepare_split(old, split)
        inputs["enrollment"] = expanded_enrollment(
            enrollment_segments[split],
            split,
            inputs["speakers"],
            config["counts_per_vowel"],
            config["enrollment_seed"],
        )
        previous = read_rows(phase6 / f"inputs/{split}/enrollment.jsonl")
        if [r for r in inputs["enrollment"] if r["enrollment_count"] == 10] != [
            r for r in previous if r["enrollment_count"] == 10
        ] or inputs["queries"] != read_rows(phase6 / f"inputs/{split}/queries.jsonl"):
            raise ValueError("baseline enrollment or queries changed")
        inputs["sources"] = {
            r["source_file"]: r for r in original_sources if r["split"] == split
        }
        group_counts = Counter(
            (s.speaker_id, s.vowel) for s in enrollment_segments[split]
        )
        inputs["available_registration_minimum"] = min(group_counts.values())
        inputs["config"] = config
        inputs["model"] = old["model"]
        inputs["trials"] = list(
            build_trials(
                inputs["queries"],
                inputs["speakers"],
                config["counts_per_vowel"],
                config["protocol_version"],
            )
        )
        for speaker in inputs["speakers"]:
            name = f"profiles/{split}/10/{speaker}.json"
            checked(phase6 / name, execution["files"][name])
            files[str((phase6 / name).relative_to(ROOT))] = execution["files"][name]
        for source in inputs["sources"].values():
            files[source["source_file"]] = source["source_sha256"]
        prepared[split] = inputs
    directories = (
        BASE,
        ROOT / "poc/phoneme-speaker-encoder/phase3_data",
        ROOT / "poc/phoneme-speaker-encoder/phase3_train",
        ROOT / "poc/phoneme-user-registration/registration",
        ROOT / "poc/phoneme-verification/verification",
        ROOT / "poc/phoneme-verification-evaluation/evaluation",
        ROOT / "poc/phoneme-baseline-comparison",
    )
    implementation = [p for folder in directories for p in folder.glob("*.py")]
    implementation += list((BASE / "tests").glob("*.py"))
    implementation += [
        ROOT / "poc/phoneme-speaker-encoder/scripts/weighted_bootstrap.py",
        ROOT / "poc/phoneme-verification-evaluation/uv.lock",
        ROOT / "poc/phoneme-verification-evaluation/pyproject.toml",
    ]
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in implementation})
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    run.mkdir(parents=True)
    for split, inputs in prepared.items():
        write_json(run / f"{split}-inputs.json", inputs)
        files[str((run / f"{split}-inputs.json").relative_to(ROOT))] = sha256_file(
            run / f"{split}-inputs.json"
        )
    write_json(
        run / "design-freeze.json",
        {
            "status": "design_frozen_before_new_inference",
            "claim_scope": config["claim_scope"],
            "files": files,
            "runtime": runtime(),
            "config": config,
        },
    )
    return prepared


def runtime():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def frozen_inputs(run, split):
    freeze = json.loads((run / "design-freeze.json").read_text())
    if freeze["runtime"] != runtime():
        raise ValueError("frozen runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    inputs = json.loads((run / f"{split}-inputs.json").read_text())
    if inputs["split"] != split:
        raise ValueError("input split mismatch")
    if split == "test":
        evaluation = json.loads((run / "evaluation-freeze.json").read_text())
        if evaluation["status"] != "thresholds_frozen_before_new_test_inference":
            raise ValueError("requires fixed validation thresholds")
        for name, checksum in evaluation["files"].items():
            checked(run / name, checksum)
    return inputs


def infer(run, split):
    inputs = frozen_inputs(run, split)
    output = run / split
    output.mkdir(exist_ok=False)
    started = perf_counter()
    encoder = FrozenEncoder.from_bundle(ROOT / inputs["model"]["bundle"])
    snapshot = model_snapshot(encoder)
    tensors = [encoder.pipeline.mean.clone(), encoder.pipeline.std.clone()]
    audio = (ValidationAudio if split == "validation" else TestAudio)(
        inputs["sources"], 16
    )
    cache = EmbeddingCache(
        audio,
        encoder.identity,
        lambda pcm, sid: encoder.embed(torch.from_numpy(pcm.copy()), sid),
        inputs["config"]["repeats_per_unique_embedding"],
        already_unit=True,
    )
    segments = {}
    for item in inputs["enrollment"] + inputs["queries"]:
        for row in item["segments"]:
            if row["segment_id"] in segments and segments[row["segment_id"]] != row:
                raise ValueError("conflicting segment identity")
            segments[row["segment_id"]] = row

    class Adapter:
        identity, pipeline, model = encoder.identity, encoder.pipeline, encoder.model

        def embed(self, pcm, sid):
            row = segments[sid]
            name, first, last = row["source_file"], row["start_frame"], row["end_frame"]
            if not np.array_equal(pcm.numpy(), audio.slice(name, first, last)):
                raise ValueError("public API PCM differs")
            first, last = effective_window(
                first, last, inputs["sources"][name]["frame_count"]
            )
            return cache.embed(name, first, last, sid)

    adapter = Adapter()
    profiles, cost, profile_checks = {}, [], 0
    phase6 = ROOT / inputs["config"]["phase6_run"]
    for item in inputs["enrollment"]:
        speaker, count = item["user_id"], item["enrollment_count"]
        profile = register_user(
            speaker,
            [EnrollmentSegment(**r) for r in item["segments"]],
            adapter,
            audio_root=ROOT,
            minimum_segments=count,
        )
        if any(r["status"] != "used" for r in profile.data["segments"]):
            raise ValueError("registration eligibility changed")
        if count == 10:
            old = load_profile(
                phase6 / f"profiles/{split}/10/{speaker}.json", encoder=encoder
            )
            if profile.checksum != old.checksum:
                raise ValueError("baseline profile mismatch")
            profile_checks += 1
        if speaker == inputs["speakers"][0]:
            fresh = register_user(
                speaker,
                [EnrollmentSegment(**r) for r in reversed(item["segments"])],
                encoder,
                audio_root=ROOT,
                minimum_segments=count,
            )
            if fresh.checksum != profile.checksum:
                raise ValueError("cache-free registration mismatch")
        save_profile(profile, output / f"profiles/{count}/{speaker}.json")
        profiles[speaker, count] = profile
        used = describe_audio(item["segments"], inputs["sources"])
        cost.append(
            {
                "speaker_id": speaker,
                "enrollment_count": count,
                "source_wavs": used["source_count"],
                "acquired_source_seconds": used["whole_source_frames"] / 24000,
                "unique_used_seconds": used["unique_frames"]["vowel_exact"] / 24000,
                "processed_seconds": used["processed_frames"]["vowel_exact"] / 24000,
                "profile_sha256": profile.checksum,
            }
        )
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    baseline = {
        (r["query_id"], r["claimed_speaker_id"]): r
        for r in iter_rows(phase6 / f"scores/{split}.jsonl")
        if r["enrollment_count"] == 10
    }
    tested, parity, maximum_delta, slots = set(), 0, 0.0, 0
    with writer(output / "scores.jsonl.gz") as write:
        for index, item in enumerate(inputs["queries"], 1):
            query = VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**r) for r in item["segments"]),
            )
            try:
                prepared = prepare_query(
                    query, next(iter(profiles.values())), adapter, audio_root=ROOT
                )
                records, counts = prepared.records, prepared.counts
                prepared.require_unchanged_sources()
            except IncompleteVerification as exc:
                records, counts = exc.records, exc.counts
            if any(r["status"] != "used" for r in records) or counts != item["counts"]:
                raise ValueError("query eligibility changed")
            grouped = {v: [] for v in VOWELS}
            if all(counts.values()):
                for row in item["segments"]:
                    grouped[row["vowel"]].append(
                        adapter.embed(
                            torch.from_numpy(
                                audio.slice(
                                    row["source_file"],
                                    row["start_frame"],
                                    row["end_frame"],
                                )
                            ),
                            row["segment_id"],
                        )
                    )
            for trial in trials[item["query_id"]]:
                count, claimed = trial["enrollment_count"], trial["claimed_speaker_id"]
                profile = profiles[claimed, count]
                values = vowel_scores({v: profile.vector(v) for v in VOWELS}, grouped)
                sample = item["speaker_id"], item["role"], count, trial["is_genuine"]
                if sample not in tested:
                    try:
                        result = verify(profile, query, encoder, audio_root=ROOT)
                        expected = {
                            "fused": result.score,
                            **{v: result.data["vowels"][v]["score"] for v in VOWELS},
                        }
                    except IncompleteVerification:
                        expected = None
                    if values != expected:
                        raise ValueError("cache-free public verification mismatch")
                    tested.add(sample)
                if count == 10:
                    reference = baseline[item["query_id"], claimed]
                    if values != reference["scores"]:
                        raise ValueError("baseline full score mismatch")
                    if values is not None:
                        maximum_delta = max(
                            maximum_delta,
                            max(
                                abs(values[k] - reference["scores"][k]) for k in values
                            ),
                        )
                    parity += 1
                write(
                    {
                        **trial,
                        "status": "scored" if values is not None else "no_score",
                        "reason": None if values is not None else "missing_vowels",
                        "scores": values,
                        "profile_sha256": profile.checksum,
                    }
                )
                slots += 1
            if index % 100 == 0:
                print(
                    f"{split}: {index}/1200 queries, {len(cache.vectors)} embeddings",
                    flush=True,
                )
    require_unchanged(encoder, snapshot)
    if not all(
        torch.equal(a, b)
        for a, b in zip(
            tensors, [encoder.pipeline.mean, encoder.pipeline.std], strict=True
        )
    ):
        raise ValueError("feature tensors changed")
    audio.require_unchanged()
    cache.save(output)
    write_json(output / "registration-cost.json", {"profiles": cost})
    write_json(
        output / "report.json",
        {
            "status": "completed",
            "profiles": len(profiles),
            "score_slots": slots,
            "baseline_profiles_identical": profile_checks,
            "baseline_scores_identical": parity,
            "maximum_baseline_score_difference": maximum_delta,
            "cache_free_verification_checks": len(tested),
            "cache_free_registration_checks": len(inputs["config"]["counts_per_vowel"]),
            "unique_embeddings": len(cache.vectors),
            "embedding_repeats": 3,
            "all_repeats_bitwise_equal": True,
            "weights_statistics_and_sources_unchanged": True,
            "elapsed_seconds": perf_counter() - started,
            "outputs_sha256": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    frozen_inputs(run, split)


def score_groups(run, split):
    groups = defaultdict(list)
    inputs = json.loads((run / f"{split}-inputs.json").read_text())
    report = json.loads((run / split / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("requires completed inference")
    for name, checksum in report["outputs_sha256"].items():
        checked(run / split / name, checksum)
    expected = {r["trial_id"]: r for r in inputs["trials"]}
    seen = set()
    for row in iter_rows(run / split / "scores.jsonl.gz"):
        trial = expected.get(row["trial_id"])
        if (
            trial is None
            or row["trial_id"] in seen
            or any(row[k] != v for k, v in trial.items())
        ):
            raise ValueError("unexpected trial identity or label")
        seen.add(row["trial_id"])
        if row["status"] not in ("scored", "no_score") or (row["scores"] is None) != (
            row["status"] == "no_score"
        ):
            raise ValueError("invalid score state")
        if row["scores"] is not None and (
            set(row["scores"]) != {"fused", *VOWELS}
            or not all(np.isfinite(v) and -1 <= v <= 1 for v in row["scores"].values())
            or row["scores"]["fused"]
            != float(np.mean([row["scores"][v] for v in VOWELS], dtype=np.float64))
        ):
            raise ValueError("invalid finite fused score")
        groups[row["enrollment_count"], row["role"]].append(
            {**row, "score": row["scores"]["fused"] if row["scores"] else None}
        )
    if len(seen) != len(expected):
        raise ValueError("missing trial")
    return inputs, groups


def calibrate(run):
    inputs, groups = score_groups(run, "validation")
    thresholds, cells, curves = {}, {}, {}
    for count in inputs["config"]["counts_per_vowel"]:
        threshold = calibrate_cell(groups[count, "verification"], inputs["speakers"])
        if threshold["status"] != "calibrated":
            raise ValueError("validation support insufficient")
        thresholds[f"n{count}"] = threshold
        for role in inputs["config"]["roles"]:
            cells[f"n{count}/{role}"], curves[f"n{count}/{role}"] = evaluate_cell(
                groups[count, role], threshold, inputs["speakers"]
            )
    old = json.loads(
        (
            ROOT
            / inputs["config"]["phase7_validation_run"]
            / "validation-thresholds.json"
        ).read_text()
    )["conditions"]["native/vowel_exact/n10/full"]["operating_points"]
    if thresholds["n10"]["operating_points"] != old:
        raise ValueError("baseline validation thresholds differ from Phase 7")
    write_json(
        run / "validation-thresholds.json",
        {"split": "validation", "role": "verification", "conditions": thresholds},
    )
    write_json(run / "validation-metrics.json", {"conditions": cells, "curves": curves})
    indices, counts = speaker_draws(
        15, inputs["config"]["bootstrap_replicates"], inputs["config"]["bootstrap_seed"]
    )
    old_counts = np.load(
        ROOT / inputs["config"]["phase7_test_run"] / "bootstrap-counts.npy",
        allow_pickle=False,
    )
    if not np.array_equal(counts, old_counts):
        raise ValueError("bootstrap draws changed")
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    np.save(run / "bootstrap-indices.npy", indices, allow_pickle=False)
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_new_test_inference",
            "new_test_inference_performed": False,
            "test_previously_observed_in_Phase3_6_7": True,
            "files": {
                name: sha256_file(run / name)
                for name in (
                    "validation-thresholds.json",
                    "validation-metrics.json",
                    "validation/report.json",
                    "bootstrap-counts.npy",
                    "bootstrap-indices.npy",
                )
            },
        },
    )


def paired(cells, replicas):
    result, arrays = {}, {}
    for role in ("verification", "cross_text_verification"):
        for count, reference in ((20, 10), (30, 10), (30, 20)):
            a, b = f"n{count}/{role}", f"n{reference}/{role}"
            for point, _ in POINTS:
                for metric in ("far", "frr", "all_input_far", "all_input_frr"):
                    name = f"{role}/n{count}_minus_n{reference}/{point}/{metric}"
                    delta = 100 * (
                        replicas[a][f"{point}/{metric}"]
                        - replicas[b][f"{point}/{metric}"]
                    )
                    arrays[name] = delta
                    result[name] = {
                        "difference_percentage_points": 100
                        * (
                            cells[a]["operating_points"][point][metric]
                            - cells[b]["operating_points"][point][metric]
                        ),
                        "ci95_percentage_points": interval(delta),
                        "population": "same query cohort and shared speaker draws; thresholds fixed per count",
                    }
    return result, arrays


def audit_rates(rows, cells, thresholds):
    checks = 0
    for (count, role), records in rows.items():
        for point, operating in thresholds[f"n{count}"]["operating_points"].items():
            tau = numeric_threshold(operating["threshold"])
            g = [r for r in records if r["is_genuine"]]
            i = [r for r in records if not r["is_genuine"]]
            sg = [r for r in g if r["status"] == "scored"]
            si = [r for r in i if r["status"] == "scored"]
            fa = sum(r["score"] >= tau for r in si)
            fr = sum(r["score"] < tau for r in sg)
            expected = {
                "false_accepts": fa,
                "false_rejects": fr,
                "all_genuine": len(g),
                "all_impostor": len(i),
                "no_score_genuine": len(g) - len(sg),
                "no_score_impostor": len(i) - len(si),
                "all_input_far": fa / len(i),
                "all_input_frr": (fr + len(g) - len(sg)) / len(g),
                "far": fa / len(si),
                "frr": fr / len(sg),
                "threshold": operating["threshold"],
            }
            measured = cells[f"n{count}/{role}"]["operating_points"][point]
            if any(measured[k] != v for k, v in expected.items()):
                raise ValueError("independent raw-score rate audit failed")
            checks += 1
    return checks


def evaluate(run):
    frozen_inputs(run, "test")
    inputs, groups = score_groups(run, "test")
    thresholds = json.loads((run / "validation-thresholds.json").read_text())[
        "conditions"
    ]
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, curves, replicas = {}, {}, {}
    for (count, role), rows in sorted(groups.items()):
        key = f"n{count}/{role}"
        cells[key], curves[key] = evaluate_cell(
            rows, thresholds[f"n{count}"], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = bootstrap_cell(
            rows, cells[key], thresholds[f"n{count}"], inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    old = json.loads(
        (ROOT / inputs["config"]["phase7_test_run"] / "test-metrics.json").read_text()
    )["conditions"]
    for role in inputs["config"]["roles"]:
        reference = old[f"native/vowel_exact/n10/full/{role}"]
        for field in (
            "queries",
            "scored_queries",
            "query_coverage",
            "pooled_eer",
            "operating_points",
            "ci95",
        ):
            if cells[f"n10/{role}"][field] != reference[field]:
                raise ValueError(f"Phase 7 baseline metric/CI parity failed: {field}")
    differences, difference_arrays = paired(cells, replicas)
    with (run / "bootstrap-metrics.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                f"{key}/{name}": value
                for key, items in replicas.items()
                for name, value in items.items()
            },
        )
    with (run / "bootstrap-paired-differences.npz").open("xb") as stream:
        np.savez_compressed(stream, **difference_arrays)
    validation = json.loads((run / "validation-metrics.json").read_text())
    _, validation_rows = score_groups(run, "validation")
    audits = audit_rates(groups, cells, thresholds) + audit_rates(
        validation_rows, validation["conditions"], thresholds
    )
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "curves": curves,
            "paired_differences": differences,
            "threshold_recalibrated": False,
            "bootstrap_replicates": 10000,
            "baseline_metric_and_ci_parity_roles": 2,
            "independent_rate_audits": audits,
        },
    )
    frozen_inputs(run, "test")


def configure_runtime():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def audit_intervals(run):
    metrics = json.loads((run / "test-metrics.json").read_text())
    checks = 0
    with np.load(run / "bootstrap-metrics.npz", allow_pickle=False) as arrays:
        for key, cell in metrics["conditions"].items():
            for name in ("query_coverage", "pooled_eer"):
                if interval(arrays[f"{key}/{name}"]) != cell["ci95"][name]:
                    raise ValueError("CI percentile audit failed")
                checks += 1
            for point, measures in cell["ci95"]["operating_points"].items():
                for name, bounds in measures.items():
                    if interval(arrays[f"{key}/{point}/{name}"]) != bounds:
                        raise ValueError("rate CI percentile audit failed")
                    checks += 1
        with np.load(
            run / "bootstrap-paired-differences.npz", allow_pickle=False
        ) as paired_arrays:
            for key, difference in metrics["paired_differences"].items():
                role, comparison, point, name = key.split("/")
                a, b = comparison.split("_minus_")
                expected = 100 * (
                    arrays[f"{a}/{role}/{point}/{name}"]
                    - arrays[f"{b}/{role}/{point}/{name}"]
                )
                if (
                    not np.array_equal(paired_arrays[key], expected)
                    or interval(expected) != difference["ci95_percentage_points"]
                ):
                    raise ValueError("paired CI audit failed")
                checks += 1
    return checks
