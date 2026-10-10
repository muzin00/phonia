"""Select identical-size, identical-count phone tokens before inference."""

import json
import wave
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from support import PHONES, ROOT, checked, core, pin, read_json, write_json


def fixed_slice(samples, first, last, frames, minimum_dbfs):
    if not 0 <= first < last <= len(samples):
        raise ValueError("invalid labeled source bounds")
    if last - first < frames:
        return None
    start = first + (last - first - frames) // 2
    rms = float(np.sqrt(np.mean(samples[start : start + frames] ** 2)))
    if rms < 10 ** (minimum_dbfs / 20):
        return None
    return start, start + frames, rms


def extract(source, meta, config, files, qc):
    path = (
        ROOT
        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
        / meta["speaker_id"]
        / f"{meta['utterance_id']}.json"
    )
    raw = read_json(path)
    if any(
        raw[k] != meta[k]
        for k in (
            "utterance_id",
            "speaker_id",
            "source_file",
            "source_sha256",
            "raw_phonemes",
        )
    ):
        raise ValueError("alignment/source metadata mismatch")
    checked(ROOT / source, meta["source_sha256"])
    pin(files, path)
    pin(files, ROOT / source)
    with wave.open(str(ROOT / source), "rb") as audio:
        if (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getcomptype(),
        ) != (config["sample_rate"], 1, 2, "NONE"):
            raise ValueError("unexpected source PCM format")
        samples = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                np.float64
            )
            / 32768
        )
    rows = []
    for index, interval in enumerate(raw["intervals"]):
        phone = interval["phoneme"]
        if phone not in PHONES:
            continue
        first, last = (
            round(interval[k] * config["sample_rate"]) for k in ("start_sec", "end_sec")
        )
        prefix = f"{meta['evaluation_role']}/{phone}"
        qc[f"{prefix}/raw"] += 1
        selected = fixed_slice(
            samples, first, last, config["segment_frames"], config["minimum_rms_dbfs"]
        )
        if selected is None:
            qc[f"{prefix}/excluded_short_or_quiet"] += 1
            continue
        start, end, rms = selected
        qc[f"{prefix}/eligible"] += 1
        origin = f"{meta['utterance_id']}--phone-{index:03d}-{phone}"
        rows.append(
            {
                "segment_id": f"{origin}--center-{start}-{end - start}",
                "original_segment_id": origin,
                "vowel": phone,
                "source_file": source,
                "source_sha256": meta["source_sha256"],
                "start_frame": start,
                "end_frame": end,
                "original_start_frame": first,
                "original_end_frame": last,
                "selected_rms_dbfs": float(20 * np.log10(rms)),
            }
        )
    return rows


def prepare_split(config, split, metadata, files, training_hashes, training_speakers):
    reference = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
    support_path = ROOT / config["source_support"] / f"{split}-inputs.json"
    original, support = read_json(reference), read_json(support_path)
    pin(files, reference)
    pin(files, support_path)
    speakers = original["speakers"]
    if speakers != support["speakers"] or set(speakers) & training_speakers:
        raise ValueError("speaker support/train separation mismatch")
    cache, qc = {}, Counter()

    def candidates(source, speaker, role):
        meta = metadata[source]
        if (meta["split"], meta["speaker_id"], meta["evaluation_role"]) != (
            split,
            speaker,
            role,
        ) or meta["source_sha256"] in training_hashes:
            raise ValueError("source role/speaker/train content leak")
        if source not in cache:
            cache[source] = extract(source, meta, config, files, qc)
        return cache[source]

    profiles = {}
    enrollment_hashes = set()
    for item in original["enrollment"]:
        speaker = item["user_id"]
        sources = sorted({row["source_file"] for row in item["segments"]})
        pool = [
            r for source in sources for r in candidates(source, speaker, "enrollment")
        ]
        selected = {
            phone: core.prior.diversify(
                [r for r in pool if r["vowel"] == phone],
                config["enrollment_segments_per_phone"],
                config["selection_seed"],
            )
            for phone in PHONES
        }
        profiles[speaker] = {
            "candidate_source_files": sources,
            "conditions": selected,
            "budget_frames": config["segment_frames"]
            * config["enrollment_segments_per_phone"],
        }
        enrollment_hashes.update(
            metadata[source]["source_sha256"] for source in sources
        )
    query_lookup = {q["query_id"]: q for q in original["queries"]}
    queries, excluded, available = [], Counter(), Counter()
    for fixed in support["queries"]:
        item = query_lookup[fixed["query_id"]]
        for name in ("speaker_id", "role", "split", "source_file", "source_sha256"):
            if item[name] != fixed[name]:
                raise ValueError("frozen source support changed")
        if item["source_sha256"] in enrollment_hashes:
            raise ValueError("enrollment/query content overlap")
        pool = candidates(item["source_file"], item["speaker_id"], item["role"])
        present = {r["vowel"] for r in pool}
        available.update(f"{item['role']}/{p}" for p in present)
        if present != set(PHONES):
            missing = "".join(p for p in PHONES if p not in present)
            excluded[f"{item['role']}/missing_fixed_window_{missing}"] += 1
            continue
        selected = {
            phone: core.prior.diversify(
                [r for r in pool if r["vowel"] == phone],
                config["query_segments_per_phone"],
                config["selection_seed"],
            )
            for phone in PHONES
        }
        queries.append(
            {
                **{
                    k: item[k]
                    for k in (
                        "query_id",
                        "utterance_id",
                        "speaker_id",
                        "source_file",
                        "source_sha256",
                        "role",
                        "split",
                    )
                },
                "conditions": selected,
                "common8": True,
                "budget_frames": config["segment_frames"]
                * config["query_segments_per_phone"],
            }
        )
    for role in core.ROLES:
        if {q["speaker_id"] for q in queries if q["role"] == role} != set(speakers):
            raise ValueError("a speaker has no eligible queries for a role")
    return {
        "split": split,
        "speakers": speakers,
        "profiles": profiles,
        "queries": queries,
        "preparation": {
            "roles": dict(Counter(q["role"] for q in queries)),
            "prior_common8_roles": support["preparation"]["roles"],
            "excluded_query_reasons": dict(excluded),
            "available_phone_queries": dict(available),
            "token_qc": dict(qc),
            "selection_uses_embeddings": False,
        },
    }


def validate_plans(inputs, config):
    enrollment_hashes = {
        row["source_sha256"]
        for item in inputs["profiles"].values()
        for rows in item["conditions"].values()
        for row in rows
    }
    plans, slices = 0, 0
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        query = "query_id" in item
        count = (
            config["query_segments_per_phone"]
            if query
            else config["enrollment_segments_per_phone"]
        )
        if (
            set(item["conditions"]) != set(PHONES)
            or item["budget_frames"] != count * config["segment_frames"]
        ):
            raise ValueError("wrong support or matched budget")
        ranges = defaultdict(list)
        for phone, rows in item["conditions"].items():
            if (
                len(rows) != count
                or len({r["original_segment_id"] for r in rows}) != count
            ):
                raise ValueError("wrong token count or repeated interval")
            for row in rows:
                first, last = row["original_start_frame"], row["original_end_frame"]
                start, end = row["start_frame"], row["end_frame"]
                if (
                    row["vowel"] != phone
                    or end - start != config["segment_frames"]
                    or first < 0
                    or start != first + (last - first - (end - start)) // 2
                    or not first <= start < end <= last
                    or row["selected_rms_dbfs"] < config["minimum_rms_dbfs"]
                ):
                    raise ValueError("wrong phone, crop, duration or quality")
                if query and (
                    row["source_sha256"] in enrollment_hashes
                    or row["source_file"] != item["source_file"]
                ):
                    raise ValueError("query source/content leak")
                if (
                    not query
                    and row["source_file"] not in item["candidate_source_files"]
                ):
                    raise ValueError("enrollment candidate source changed")
                ranges[row["source_file"]].append((start, end))
                slices += 1
            plans += 1
        if any(
            a[1] > b[0] for rows in ranges.values() for a, b in pairwise(sorted(rows))
        ):
            raise ValueError("selected phone intervals overlap")
    return {"plans": plans, "slices": slices}


def prepare_inputs(config, run):
    if run.exists():
        raise ValueError("unused output directory required")
    files = {}
    source_freeze = ROOT / config["source_support"] / "design-freeze.json"
    for name, sha in read_json(source_freeze)["files"].items():
        checked(ROOT / name, sha)
        files[name] = sha
    pin(files, source_freeze)
    path = ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    metadata = {r["source_file"]: r for r in core.old.previous.read_rows(path)}
    pin(files, path)
    train_manifest = ROOT / config["training_run"] / "training-segments.jsonl"
    training_hashes, training_speakers = set(), set()
    with train_manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            training_hashes.add(row["source_sha256"])
            training_speakers.add(row["speaker_id"])
    pin(files, train_manifest)
    run.mkdir(parents=True)
    for split in ("validation", "test"):
        inputs = prepare_split(
            config, split, metadata, files, training_hashes, training_speakers
        )
        audit = validate_plans(inputs, config)
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
        print(f"{split}: {inputs['preparation']['roles']}; {audit}", flush=True)
    return files
