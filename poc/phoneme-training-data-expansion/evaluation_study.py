"""Two frozen encoders, identical enrollment/queries and validation-only thresholds."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-enrollment-scaling"))
import numpy as np
import scaling as previous
import torch
from final_metrics import bootstrap_cell, interval
from phase3_data.input import InputPipeline
from phase3_data.manifest import json_sha256
from phase3_train.models import create_encoder
from registration import EnrollmentSegment, FrozenEncoder, load_profile
from validation import iter_rows, writer
from validation_metrics import POINTS, calibrate_cell, evaluate_cell
from weighted_bootstrap import speaker_draws

ROOT = previous.ROOT
VOWELS = previous.VOWELS
MODELS = ("jvs70", "jvs70_cv70")
ROLES = ("verification", "cross_text_verification")
METRICS = ("far", "frr", "all_input_far", "all_input_frr")
checked, sha256_file, write_json = (
    previous.checked,
    previous.sha256_file,
    previous.write_json,
)


def read_json(path):
    return json.loads(path.read_text())


def pin(files, path):
    files[str(path.relative_to(ROOT))] = sha256_file(path)


def prepare(run):
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused output directory inside artifacts required")
    protocol_path = BASE / "config/evaluation-protocol.json"
    config = read_json(protocol_path)
    if (
        config["conditions"] != list(MODELS)
        or config["enrollment_count_per_vowel"] != 10
        or config["required_vowels"] != list(VOWELS)
        or config["roles"] != list(ROLES)
        or config["query_window"] != "full"
        or config["calibration"] != "pooled_validation_verification_per_model"
        or config["operating_points"] != [p for p, _ in POINTS]
        or config["acceptance"] != "score_gte_threshold"
        or config["no_score"] != "reject_and_keep_in_all_input_denominators"
        or config["fusion"]
        != "mean_segment_cosine_within_vowel_then_equal_mean_over_five_vowels"
        or config["retrain_encoder"]
        or config["update_feature_statistics"]
        or config["test_threshold_recalibration"]
        or config["independent_holdout"]
        or not config["reuse_baseline_scores"]
        or config["repeats_per_unique_embedding"] != 3
        or config["bootstrap_replicates"] != 10000
        or config["bootstrap_seed"] != 20260929
    ):
        raise ValueError("unsupported evaluation protocol")
    prior = ROOT / config["prior_run"]
    checked(prior / "study-report.json", config["prior_report_sha256"])
    prior_report = read_json(prior / "study-report.json")
    prior_freeze = read_json(prior / "design-freeze.json")
    if (
        prior_report["status"] != "completed"
        or prior_freeze["runtime"] != previous.runtime()
    ):
        raise ValueError("prior study incomplete or runtime changed")
    files = dict(prior_freeze["files"])
    for name, checksum in prior_report["outputs_sha256"].items():
        checked(prior / name, checksum)
        files[str((prior / name).relative_to(ROOT))] = checksum
    pin(files, prior / "study-report.json")
    training = ROOT / config["training_run"]
    summary_path = training / "expanded/training/summary.json"
    checked(summary_path, config["training_summary_sha256"])
    summary = read_json(summary_path)
    if summary["status"] != "completed" or summary["test_used"]:
        raise ValueError("requires completed training selected without test")
    for name, checksum in summary["outputs_sha256"].items():
        checked(training / "expanded" / name, checksum)
        files[str((training / "expanded" / name).relative_to(ROOT))] = checksum
    for path in (summary_path, training / "training-freeze.json", protocol_path):
        pin(files, path)
    prepared = {}
    for split in ("validation", "test"):
        old = read_json(prior / f"{split}-inputs.json")
        inputs = {
            k: old[k] for k in ("split", "speakers", "queries", "sources", "model")
        }
        inputs["enrollment"] = [
            e for e in old["enrollment"] if e["enrollment_count"] == 10
        ]
        inputs["config"] = config
        inputs["trials"] = list(
            previous.build_trials(
                inputs["queries"], inputs["speakers"], [10], config["protocol_version"]
            )
        )
        validate_population(inputs)
        prepared[split] = inputs
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    split_labels = read_json(split_path)["speaker_splits"]
    pin(files, split_path)
    if set(prepared["validation"]["speakers"]) & set(prepared["test"]["speakers"]):
        raise ValueError("validation/test speaker overlap")
    for split, inputs in prepared.items():
        if set(inputs["speakers"]) != set(split_labels[split]) or set(
            inputs["speakers"]
        ) & set(split_labels["train"]):
            raise ValueError("JVS speaker split changed")
    paths = [
        BASE / name
        for name in ("evaluation_study.py", "evaluation_report.py", "run_evaluation.py")
    ]
    paths += list((BASE / "tests").glob("test_evaluation.py"))
    for path in paths:
        pin(files, path)
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    run.mkdir(parents=True)
    for split, inputs in prepared.items():
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "design_frozen_before_new_inference",
            "config": config,
            "files": files,
            "runtime": previous.runtime(),
            "cross_corpus_person_overlap": "unknown; no_reidentification_attempted",
        },
    )


def validate_population(inputs):
    speakers = inputs["speakers"]
    if len(speakers) != 15 or len(set(speakers)) != 15:
        raise ValueError("requires original 15 speakers")
    if len(inputs["enrollment"]) != 15 or len(inputs["queries"]) != 1200:
        raise ValueError("unexpected enrollment/query population")
    enrollment_hashes = set()
    for item in inputs["enrollment"]:
        if Counter(s["vowel"] for s in item["segments"]) != Counter(
            {v: 10 for v in VOWELS}
        ):
            raise ValueError("changed enrollment count")
        enrollment_hashes.update(s["source_sha256"] for s in item["segments"])
    if any(q["source_sha256"] in enrollment_hashes for q in inputs["queries"]):
        raise ValueError("enrollment/query content overlap")
    if len(inputs["trials"]) != 18000:
        raise ValueError("incomplete trial matrix")


def frozen_inputs(run, split):
    if split not in ("validation", "test"):
        raise ValueError("validation/test only")
    freeze = read_json(run / "design-freeze.json")
    if (
        freeze["status"] != "design_frozen_before_new_inference"
        or freeze["runtime"] != previous.runtime()
    ):
        raise ValueError("frozen design/runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    inputs = read_json(run / f"{split}-inputs.json")
    if inputs["split"] != split:
        raise ValueError("input split mismatch")
    if split == "test":
        evaluation = read_json(run / "evaluation-freeze.json")
        if evaluation["status"] != "thresholds_frozen_before_new_test_inference":
            raise ValueError("fixed validation thresholds required")
        for name, checksum in evaluation["files"].items():
            checked(run / name, checksum)
    return inputs


def expanded_encoder(config):
    output = ROOT / config["training_run"] / "expanded"
    summary = read_json(output / "training/summary.json")
    checked(output / "training/summary.json", config["training_summary_sha256"])
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json", "run.json"):
        checked(output / name, summary["outputs_sha256"][name])
    run = read_json(output / "run.json")
    export = torch.load(
        output / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if (
        export["checkpoint_update"] != summary["selected_update"]
        or export["encoder"] != "statistics_mlp"
        or export["embedding_dimension"] != 128
        or export["encoder_parameters"] != 65920
        or export["run_seed"] != 20260926
        or export["configuration_sha256"] != run["configuration_sha256"]
        or json_sha256(run["settings"]) != run["configuration_sha256"]
        or run["settings"]["cohort"] != 140
    ):
        raise ValueError("expanded inference bundle metadata mismatch")
    phase3 = ROOT / "poc/phoneme-speaker-encoder"
    baseline = read_json(phase3 / "config/baseline-log-mel.json")
    statistics = read_json(output / "bundle/feature-statistics.json")
    if statistics["speaker_count"] != 140 or statistics["cohort"] != 140:
        raise ValueError("expanded statistics population mismatch")
    pipeline = InputPipeline(
        baseline["input"], rms_enabled=False, statistics=statistics
    )
    model = create_encoder("statistics_mlp")
    model.load_state_dict(export["model"], strict=True)
    identity = {
        "design_version": "2.0.0",
        "encoder": "statistics_mlp",
        "seed": 20260926,
        "checkpoint_sha256": sha256_file(output / "bundle/encoder.pt"),
        "encoder_config_sha256": json_sha256(
            {
                name: sha256_file(phase3 / "config" / name)
                for name in ("log-mel-encoders.json", "waveform-encoders.json")
            }
        ),
        "feature_statistics_sha256": sha256_file(
            output / "bundle/feature-statistics.json"
        ),
        "preprocessing_sha256": pipeline.preprocessing_sha256,
        "implementation_sha256": json_sha256(
            {
                name: sha256_file(phase3 / name)
                for name in ("phase3_data/input.py", "phase3_train/models.py")
            }
        ),
    }
    return FrozenEncoder(model, pipeline, identity)


def reuse_baseline(run, split, inputs):
    prior = ROOT / inputs["config"]["prior_run"]
    output = run / split / "jvs70"
    output.mkdir(parents=True, exist_ok=False)
    report = read_json(prior / split / "report.json")
    if (
        not report["all_repeats_bitwise_equal"]
        or not report["weights_statistics_and_sources_unchanged"]
    ):
        raise ValueError("baseline inference not verified")
    reference = {
        (r["query_id"], r["claimed_speaker_id"]): r
        for r in iter_rows(prior / split / "scores.jsonl.gz")
        if r["enrollment_count"] == 10
    }
    if len(reference) != len(inputs["trials"]):
        raise ValueError("baseline trial population mismatch")
    with writer(output / "scores.jsonl.gz") as write:
        for trial in inputs["trials"]:
            old = reference[trial["query_id"], trial["claimed_speaker_id"]]
            for key in (
                "query_id",
                "split",
                "role",
                "speaker_id",
                "claimed_speaker_id",
                "enrollment_count",
                "is_genuine",
            ):
                if old[key] != trial[key]:
                    raise ValueError("baseline trial label changed")
            write(
                {
                    **trial,
                    "model_id": "jvs70",
                    **{
                        key: old[key]
                        for key in ("status", "reason", "scores", "profile_sha256")
                    },
                }
            )
    write_json(
        output / "report.json",
        {
            "status": "completed",
            "reused": True,
            "source": str((prior / split).relative_to(ROOT)),
            "baseline_score_parity": len(reference),
            "source_report_sha256": sha256_file(prior / split / "report.json"),
            "outputs_sha256": {
                "scores.jsonl.gz": sha256_file(output / "scores.jsonl.gz")
            },
        },
    )


def infer_expanded(run, split, inputs):
    output = run / split / "jvs70_cv70"
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    encoder = expanded_encoder(inputs["config"])
    snapshot = previous.model_snapshot(encoder)
    tensors = [encoder.pipeline.mean.clone(), encoder.pipeline.std.clone()]
    audio = (previous.ValidationAudio if split == "validation" else previous.TestAudio)(
        inputs["sources"], 16
    )
    cache = previous.EmbeddingCache(
        audio,
        encoder.identity,
        lambda pcm, sid: encoder.embed(torch.from_numpy(pcm.copy()), sid),
        3,
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
            first, last = previous.effective_window(
                first, last, inputs["sources"][name]["frame_count"]
            )
            return cache.embed(name, first, last, sid)

    adapter = Adapter()
    profiles = {}
    for item in inputs["enrollment"]:
        speaker = item["user_id"]
        profile = previous.register_user(
            speaker,
            [EnrollmentSegment(**r) for r in item["segments"]],
            adapter,
            audio_root=ROOT,
        )
        if any(r["status"] != "used" for r in profile.data["segments"]):
            raise ValueError("registration eligibility changed")
        if speaker == inputs["speakers"][0]:
            fresh = previous.register_user(
                speaker,
                [EnrollmentSegment(**r) for r in reversed(item["segments"])],
                encoder,
                audio_root=ROOT,
            )
            if fresh.checksum != profile.checksum:
                raise ValueError("cache-free registration mismatch")
        path = output / "profiles" / f"{speaker}.json"
        previous.save_profile(profile, path)
        profiles[speaker] = load_profile(path, encoder=encoder)
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    tested, slots = set(), 0
    with writer(output / "scores.jsonl.gz") as write:
        for index, item in enumerate(inputs["queries"], 1):
            query = previous.VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**r) for r in item["segments"]),
            )
            try:
                prepared = previous.prepare_query(
                    query, next(iter(profiles.values())), adapter, audio_root=ROOT
                )
                records, counts = prepared.records, prepared.counts
                prepared.require_unchanged_sources()
            except previous.IncompleteVerification as exc:
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
                profile = profiles[trial["claimed_speaker_id"]]
                values = previous.vowel_scores(
                    {v: profile.vector(v) for v in VOWELS}, grouped
                )
                sample = (
                    item["speaker_id"],
                    item["role"],
                    tuple(v for v in VOWELS if counts[v]),
                    trial["is_genuine"],
                )
                if sample not in tested:
                    try:
                        result = previous.verify(
                            profile, query, encoder, audio_root=ROOT
                        )
                        expected = {
                            "fused": result.score,
                            **{v: result.data["vowels"][v]["score"] for v in VOWELS},
                        }
                    except previous.IncompleteVerification:
                        expected = None
                    if values != expected:
                        raise ValueError("cache-free public verification mismatch")
                    tested.add(sample)
                write(
                    {
                        **trial,
                        "model_id": "jvs70_cv70",
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
    previous.require_unchanged(encoder, snapshot)
    if not all(
        torch.equal(a, b)
        for a, b in zip(
            tensors, [encoder.pipeline.mean, encoder.pipeline.std], strict=True
        )
    ):
        raise ValueError("feature tensors changed")
    audio.require_unchanged()
    cache.save(output)
    write_json(
        output / "report.json",
        {
            "status": "completed",
            "reused": False,
            "profiles": len(profiles),
            "score_slots": slots,
            "encoder_identity": encoder.identity,
            "cache_free_verification_checks": len(tested),
            "cache_free_registration_checks": 1,
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


def infer(run, split):
    inputs = frozen_inputs(run, split)
    reuse_baseline(run, split, inputs)
    infer_expanded(run, split, inputs)
    frozen_inputs(run, split)


def score_groups(run, split):
    inputs = read_json(run / f"{split}-inputs.json")
    expected = {r["trial_id"]: r for r in inputs["trials"]}
    groups = defaultdict(list)
    status_by_model = {}
    for model in MODELS:
        output = run / split / model
        report = read_json(output / "report.json")
        if report["status"] != "completed":
            raise ValueError("incomplete inference")
        for name, checksum in report["outputs_sha256"].items():
            checked(output / name, checksum)
        seen = {}
        for row in iter_rows(output / "scores.jsonl.gz"):
            trial = expected.get(row["trial_id"])
            if (
                trial is None
                or row["trial_id"] in seen
                or row["model_id"] != model
                or any(row[k] != v for k, v in trial.items())
            ):
                raise ValueError("unexpected trial identity/label/model")
            if row["status"] not in ("scored", "no_score") or (
                row["scores"] is None
            ) != (row["status"] == "no_score"):
                raise ValueError("invalid score state")
            values = row["scores"]
            if values is not None and (
                set(values) != {"fused", *VOWELS}
                or not all(np.isfinite(v) and -1 <= v <= 1 for v in values.values())
                or values["fused"]
                != float(np.mean([values[v] for v in VOWELS], dtype=np.float64))
            ):
                raise ValueError("invalid finite fused score")
            seen[row["trial_id"]] = row["status"]
            groups[model, row["role"]].append(
                {**row, "score": values["fused"] if values else None}
            )
        if set(seen) != set(expected):
            raise ValueError("missing trial")
        status_by_model[model] = seen
    if status_by_model[MODELS[0]] != status_by_model[MODELS[1]]:
        raise ValueError("model comparison has different score coverage")
    return inputs, groups


def calibrate(run):
    inputs, groups = score_groups(run, "validation")
    thresholds, cells, curves = {}, {}, {}
    for model in MODELS:
        thresholds[model] = calibrate_cell(
            groups[model, "verification"], inputs["speakers"]
        )
        if thresholds[model]["status"] != "calibrated":
            raise ValueError("validation support insufficient")
        for role in ROLES:
            key = f"{model}/{role}"
            cells[key], curves[key] = evaluate_cell(
                groups[model, role], thresholds[model], inputs["speakers"]
            )
    prior = ROOT / inputs["config"]["prior_run"]
    baseline = read_json(prior / "validation-thresholds.json")["conditions"]["n10"]
    if thresholds["jvs70"]["operating_points"] != baseline["operating_points"]:
        raise ValueError("baseline threshold parity failed")
    write_json(
        run / "validation-thresholds.json",
        {"split": "validation", "role": "verification", "conditions": thresholds},
    )
    write_json(run / "validation-metrics.json", {"conditions": cells, "curves": curves})
    indices, counts = speaker_draws(15, 10000, 20260929)
    if not np.array_equal(
        counts, np.load(prior / "bootstrap-counts.npy", allow_pickle=False)
    ):
        raise ValueError("bootstrap draws changed")
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    np.save(run / "bootstrap-indices.npy", indices, allow_pickle=False)
    names = [
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-counts.npy",
        "bootstrap-indices.npy",
    ]
    names += [f"validation/{model}/report.json" for model in MODELS]
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_new_test_inference",
            "new_test_inference_performed": False,
            "test_previously_observed": True,
            "files": {name: sha256_file(run / name) for name in names},
        },
    )


def metric_value(cell, name):
    if "/" not in name:
        return cell[name]
    point, metric = name.split("/")
    return cell["operating_points"][point][metric]


def paired(cells, replicas):
    differences, arrays = {}, {}
    for role in ROLES:
        a, b = f"jvs70_cv70/{role}", f"jvs70/{role}"
        names = [f"{p}/{m}" for p, _ in POINTS for m in METRICS]
        names += ["pooled_eer", "query_coverage"]
        for name in names:
            key = f"{role}/{name}"
            delta = 100 * (replicas[a][name] - replicas[b][name])
            arrays[key] = delta

            differences[key] = {
                "difference_percentage_points": 100
                * (metric_value(cells[a], name) - metric_value(cells[b], name)),
                "ci95_percentage_points": interval(delta),
                "comparison": "expanded_minus_baseline; same queries and shared speaker draws; model-specific fixed validation thresholds",
            }
    return differences, arrays


def evaluate(run):
    frozen_inputs(run, "test")
    inputs, groups = score_groups(run, "test")
    thresholds = read_json(run / "validation-thresholds.json")["conditions"]
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, curves, replicas = {}, {}, {}
    for (model, role), rows in sorted(groups.items()):
        key = f"{model}/{role}"
        cells[key], curves[key] = evaluate_cell(
            rows, thresholds[model], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = bootstrap_cell(
            rows, cells[key], thresholds[model], inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    prior = ROOT / inputs["config"]["prior_run"]
    old = read_json(prior / "test-metrics.json")["conditions"]
    for role in ROLES:
        for field in (
            "queries",
            "scored_queries",
            "query_coverage",
            "pooled_eer",
            "operating_points",
            "ci95",
        ):
            if cells[f"jvs70/{role}"][field] != old[f"n10/{role}"][field]:
                raise ValueError(f"baseline metric/CI parity failed: {field}")
    differences, arrays = paired(cells, replicas)
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{
            f"{key}/{name}": value
            for key, items in replicas.items()
            for name, value in items.items()
        },
    )
    np.savez_compressed(run / "bootstrap-paired-differences.npz", **arrays)
    validation_inputs, validation_groups = score_groups(run, "validation")
    validation_cells = read_json(run / "validation-metrics.json")["conditions"]
    audits = 0
    for model in MODELS:
        for rows, measures in ((groups, cells), (validation_groups, validation_cells)):
            audits += previous.audit_rates(
                {(10, role): rows[model, role] for role in ROLES},
                {f"n10/{role}": measures[f"{model}/{role}"] for role in ROLES},
                {"n10": thresholds[model]},
            )
    if (
        validation_inputs["speakers"]
        != read_json(run / "validation-inputs.json")["speakers"]
    ):
        raise ValueError("validation inputs changed")
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


def audit_intervals(run):
    metrics = read_json(run / "test-metrics.json")
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
        ) as differences:
            for key, measured in metrics["paired_differences"].items():
                role, name = key.split("/", 1)
                expected = 100 * (
                    arrays[f"jvs70_cv70/{role}/{name}"] - arrays[f"jvs70/{role}/{name}"]
                )
                if (
                    not np.array_equal(differences[key], expected)
                    or interval(expected) != measured["ci95_percentage_points"]
                ):
                    raise ValueError("paired CI audit failed")
                checks += 1
    return checks
