"""Fixed-model diagnostics: missing consonants and noise on query intervals."""

from __future__ import annotations

import copy
import importlib
import json
import wave
from collections import defaultdict

import data as shared

spec = importlib.util.spec_from_file_location(
    "fixed_fusion_study", shared.BASE / "study.py"
)
fusion_study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_study)
create_model, score = fusion_study.create_model, fusion_study.score

primitives = importlib.import_module("common")
STRESS = {
    "seed": 20261018,
    "missing_consonant_probability": 0.5,
    "noisy_query_interval_probability": 0.2,
    "white_noise_snr_db": 10,
    "threshold_policy": "original clean validation thresholds; no recalibration",
    "models": "all six already selected models; no further selection or training",
    "alignment": "original clean boundaries retained; synthetic noise added after slicing",
}


def selected(name, probability):
    return int(primitives.rank(STRESS["seed"], name)[:8], 16) / 2**32 < probability


def missing_inputs(original):
    data = copy.deepcopy(original)
    removed = 0
    for query in data["queries"]:
        for phone in query["groups"]:
            if phone not in primitives.VOWELS and selected(
                f"missing/{query['query_id']}/{phone}",
                STRESS["missing_consonant_probability"],
            ):
                removed += bool(query["groups"][phone])
                query["groups"][phone] = []
    return data, {"removed_present_query_phone_groups": removed}


def noisy_vectors(source, vectors, segments):
    result = vectors.copy()
    metadata = shared.np.array(
        [
            [(r["end_frame"] - r["start_frame"]) / 24000, r["rms_dbfs"]]
            for r in segments
        ],
        shared.np.float32,
    )
    by_source = defaultdict(list)
    for row in segments:
        if row["role"] != "enrollment" and selected(
            f"noise/{row['segment_id']}", STRESS["noisy_query_interval_probability"]
        ):
            by_source[row["source_file"]].append(row)
    pipe = primitives.pipeline(
        shared.read_json(source / "design-freeze.json")["config"]
    )
    features, indices, snrs = [], [], []
    clipped = 0
    for name, intervals in sorted(by_source.items()):
        with wave.open(str(shared.ROOT / name), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
                24000,
                1,
                2,
            ):
                raise ValueError("unexpected original WAV")
            pcm = (
                shared.np.frombuffer(
                    wav.readframes(wav.getnframes()), dtype="<i2"
                ).astype(shared.np.float32)
                / 32768
            )
        for row in intervals:
            clean = pcm[row["start_frame"] : row["end_frame"]].astype(shared.np.float64)
            rng = shared.np.random.Generator(
                shared.np.random.PCG64(
                    int(primitives.rank(STRESS["seed"], row["segment_id"])[:16], 16)
                )
            )
            noise = rng.standard_normal(len(clean))
            noise *= shared.np.sqrt(
                shared.np.mean(clean**2) / shared.np.mean(noise**2)
            ) / 10 ** (STRESS["white_noise_snr_db"] / 20)
            raw = clean + noise
            clipped += int(shared.np.count_nonzero(shared.np.abs(raw) > 1))
            corrupted = shared.np.clip(raw, -1, 1).astype(shared.np.float32)
            snrs.append(
                float(
                    10
                    * shared.np.log10(
                        shared.np.mean(clean**2)
                        / shared.np.mean((corrupted - clean) ** 2)
                    )
                )
            )
            features.append(
                primitives.pool_features(pipe, corrupted, row["segment_id"])
            )
            indices.append(row["cache_index"])
            metadata[row["cache_index"], 1] = 20 * shared.np.log10(
                shared.np.sqrt(shared.np.mean(corrupted.astype(shared.np.float64) ** 2))
            )
    model, _ = shared.load_model(
        source / "trials" / shared.read_json(shared.CONFIG)["source_trial"]
    )
    result[indices] = shared.model_embeddings(
        model, shared.np.asarray(features, shared.np.float32)
    )
    return (
        result,
        metadata,
        {
            "corrupted_intervals": len(indices),
            "query_intervals": sum(r["role"] != "enrollment" for r in segments),
            "affected_source_wavs": len(by_source),
            "clipped_samples": clipped,
            "achieved_snr_db_minimum": min(snrs),
            "achieved_snr_db_maximum": max(snrs),
            "corrupted_cache_indices": indices,
        },
    )


def run_stress(run):
    shared.torch.set_num_threads(1)
    config = shared.read_json(shared.CONFIG)
    frozen = shared.verify(run)
    selection = shared.read_json(run / "selection-freeze.json")
    for name, files in selection["models"].items():
        for file, checksum in files.items():
            shared.checked(run / name / file, checksum)
    directory = run / "stress"
    if directory.exists():
        raise ValueError("stress requires an unused output directory")
    directory.mkdir()
    shared.write_json(
        directory / "design-freeze.json",
        {
            "settings": STRESS,
            "script_sha256": shared.sha256_file(__file__),
            "primary_design_sha256": shared.sha256_file(run / "design-freeze.json"),
            "selection_sha256": shared.sha256_file(run / "selection-freeze.json"),
            "clean_test_used_only_to_trigger_planned_diagnostics": True,
            "all_challenges_fixed_before_stress_scoring": True,
        },
    )
    source = shared.ROOT / config["source_run"]
    condition = shared.ROOT / config["evaluation_run"] / "enrollment-30"
    original = shared.read_json(run / "test-inputs.json")
    segments = list(shared.rows(source / "test-segments.jsonl"))
    vectors = shared.np.load(condition / "test-embeddings.npy", allow_pickle=False)
    original_metadata = shared.np.array(
        [
            [(r["end_frame"] - r["start_frame"]) / 24000, r["rms_dbfs"]]
            for r in segments
        ],
        shared.np.float32,
    )
    missing, missing_summary = missing_inputs(original)
    print("Re-embedding query intervals with partial SNR 10dB white noise", flush=True)
    noisy, noisy_metadata, noisy_summary = noisy_vectors(source, vectors, segments)
    results = {}
    for label, inputs, embedding, metadata, summary in (
        (
            "missing-half-consonants",
            missing,
            vectors,
            original_metadata,
            missing_summary,
        ),
        ("noise-20pct-intervals-snr10", original, noisy, noisy_metadata, noisy_summary),
    ):
        output = directory / label
        output.mkdir()
        shared.write_json(output / "test-inputs.json", inputs)
        shared.np.save(output / "test-embeddings.npy", embedding, allow_pickle=False)
        baseline = shared.scores_for(inputs, embedding, frozen["phones"])
        arrays = {
            k: shared.torch.from_numpy(v)
            for k, v in shared.aggregates(
                inputs, embedding, metadata, frozen["phones"]
            ).items()
        }
        points = shared.read_json(condition / "validation-thresholds.json")[
            "thresholds"
        ]
        shared.write_rows(output / "test-scores.jsonl", baseline)
        report = {
            "phonemes": frozen["phones"],
            "embeddings_sha256": shared.sha256_file(output / "test-embeddings.npy"),
            "scores_sha256": shared.sha256_file(output / "test-scores.jsonl"),
            "metrics": shared.metrics(baseline, points),
        }
        shared.write_json(output / "test-metrics.json", report)
        checks = {"equal": shared.audit_trial(output, output, "test", report)}
        summaries = {"equal": report["metrics"]}
        verification = importlib.import_module("verification")
        for name in selection["models"]:
            bundle = shared.torch.load(
                run / name / "fusion.pt", map_location="cpu", weights_only=True
            )
            model = create_model(config, bundle["kind"])
            model.load_state_dict(bundle["state_dict"])
            values = score(model, arrays, baseline, frozen["phones"])
            points = shared.read_json(run / name / "validation-thresholds.json")[
                "thresholds"
            ]
            report = {"metrics": shared.metrics(values, points)}
            shared.write_rows(output / f"{name}-scores.jsonl", values)
            shared.write_json(output / f"{name}-metrics.json", report)
            checks[name] = verification.check_values(values, baseline, report)
            summaries[name] = report["metrics"]
        results[label] = {"conditions": summary, "variants": summaries, "audit": checks}
        print(
            json.dumps(
                {
                    "challenge": label,
                    "eer": {
                        n: {r: m["eer"] for r, m in roles.items()}
                        for n, roles in summaries.items()
                    },
                }
            ),
            flush=True,
        )
    shared.verify(run, full=True)
    shared.write_json(
        directory / "results.json", {"settings": STRESS, "challenges": results}
    )
    hashes = {
        str(p.relative_to(shared.ROOT)): shared.sha256_file(p)
        for p in directory.rglob("*")
        if p.is_file()
    }
    shared.write_json(
        directory / "completion-verification.json",
        {
            "status": "passed",
            "source_hashes_and_selected_models_verified": True,
            "output_sha256": hashes,
        },
    )


if __name__ == "__main__":
    run_stress(
        shared.ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1"
    )
