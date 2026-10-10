"""Inventory every speech phone and freeze train/validation/test source slices."""

from __future__ import annotations

import json
import math
import wave
from collections import Counter, defaultdict

from common import (
    BASE,
    CONFIG,
    ROOT,
    VOWELS,
    canonical,
    check_cache_equivalence,
    create_encoder,
    expanded,
    np,
    ordered,
    pin,
    pipeline,
    pool_features,
    rank,
    read_json,
    relative,
    review_shared,
    torch,
    write_json,
    write_rows,
)


def extract(path, meta, config, files, qc):
    alignment = read_json(path)
    if any(
        alignment[k] != meta[k]
        for k in ("utterance_id", "speaker_id", "source_file", "source_sha256")
    ):
        raise ValueError("alignment and source metadata differ")
    raw = alignment["raw_phonemes"]
    intervals = alignment["intervals"]
    if (
        len(intervals) != len(raw) + 2
        or [r["phoneme"] for r in intervals[1:-1]] != alignment["julius_phonemes"]
    ):
        raise ValueError("raw G2P labels no longer match alignment indices")
    if "raw_phonemes" in meta and raw != meta["raw_phonemes"]:
        raise ValueError("JVS transcript/phone mismatch")
    pin(files, path)
    source = ROOT / meta["source_file"]
    pin(files, source)
    if files[relative(source)] != meta["source_sha256"]:
        raise ValueError("original WAV changed")
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError("expected 24kHz mono PCM16")
        samples = (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                np.float32
            )
            / 32768
        )
    result = []
    for index, (phone, interval) in enumerate(zip(raw, intervals[1:-1], strict=True)):
        normalized = canonical(phone)
        if normalized is None:
            qc["pause"] += 1
            continue
        first, last = (
            round(interval["start_sec"] * 24000),
            round(interval["end_sec"] * 24000),
        )
        if not 0 <= first < last <= len(samples):
            raise ValueError("invalid original boundary")
        original_first, original_last = first, last
        size = min(config["maximum_frames"], last - first)
        first += (last - first - size) // 2
        last = first + size
        clip = samples[first:last]
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        reasons = []
        if size < config["minimum_frames"]:
            reasons.append("short_interval")
        if rms < 10 ** (config["minimum_rms_dbfs"] / 20):
            reasons.append("low_rms")
        qc[f"{normalized}/raw"] += 1
        for reason in reasons:
            qc[f"{normalized}/{reason}"] += 1
        identifier = f"{meta['utterance_id']}--g2p-{index:03d}-{normalized}--center-{first}-{size}"
        row = {
            "segment_id": identifier,
            "speaker_id": meta["speaker_id"],
            "phoneme": normalized,
            "raw_phoneme": phone,
            "alignment_phoneme": interval["phoneme"],
            "split": meta["split"],
            "role": meta["evaluation_role"],
            "source_file": meta["source_file"],
            "source_sha256": meta["source_sha256"],
            "start_frame": first,
            "end_frame": last,
            "raw_start_frame": original_first,
            "raw_end_frame": original_last,
            "interval_index": index + 1,
            "alignment_file": relative(path),
            "rms_dbfs": 20 * math.log10(rms) if rms else None,
            "quality_flags": reasons,
            "utterance_id": meta["utterance_id"],
        }
        result.append((row, clip))
    return result


def diverse(pool, count, seed):
    by_source = defaultdict(list)
    for row in sorted(
        pool, key=lambda r: (rank(seed, r["segment_id"]), r["segment_id"])
    ):
        by_source[row["source_file"]].append(row)
    sources = sorted(by_source, key=lambda name: (rank(seed, name), name))
    result = []
    while len(result) < count and sources:
        for source in list(sources):
            result.append(by_source[source].pop(0))
            if not by_source[source]:
                sources.remove(source)
            if len(result) == count:
                break
    return result


def prepare():
    config = read_json(CONFIG)
    config["run_directory"] = config.pop(
        "preparation_run_directory", config["run_directory"]
    )
    run = ROOT / config["run_directory"]
    if run.exists():
        raise ValueError("unused run directory required")
    torch.set_num_threads(1)
    pipe = pipeline(config)
    files = {}
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    split = read_json(split_path)["speaker_splits"]
    cv = ROOT / config["common_voice_source"]
    selection = read_json(cv / "selection.json")
    cv_speakers = sorted({r["speaker_id"] for r in selection})
    if (
        len(cv_speakers) != 70
        or len(split["train"]) != 70
        or any(r["locale"] != "ja" for r in selection)
    ):
        raise ValueError("unexpected training speaker selection")
    train_speakers = sorted([*split["train"], *cv_speakers])
    metadata, alignment_paths = {}, {}
    for corpus, meta_path, alignment_dir in (
        (
            "JVS",
            ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl",
            ROOT / "poc/phoneme-speaker-dataset/data/generated/alignments/raw",
        ),
        ("Common Voice", cv / "additional-utterances.jsonl", cv / "alignments/raw"),
    ):
        pin(files, meta_path)
        for line in meta_path.open():
            row = json.loads(line)
            row["corpus"] = corpus
            if row["source_file"] in metadata:
                raise ValueError("duplicate original source")
            if corpus == "Common Voice" and (
                row["speaker_id"] not in cv_speakers or row["source_split"] != "train"
            ):
                raise ValueError("non-training Common Voice source")
            metadata[row["source_file"]] = row
            alignment_paths[row["source_file"]] = (
                alignment_dir / row["speaker_id"] / f"{row['utterance_id']}.json"
            )
    inventory = set()
    missing_alignments = []
    for source in sorted(metadata):
        if alignment_paths[source].exists():
            labels = read_json(alignment_paths[source])["raw_phonemes"]
        else:
            labels = metadata[source].get("raw_phonemes")
            if labels is None:
                prepared = cv / "prepared" / f"{metadata[source]['utterance_id']}.json"
                pin(files, prepared)
                expected = read_json(prepared).get("expected")
                labels = expected["raw_phonemes"] if expected else []
            missing_alignments.append(relative(alignment_paths[source]))
        inventory.update(p for r in labels if (p := canonical(r)) is not None)
    if not set(VOWELS) <= inventory:
        raise ValueError("fixed vowel inventory missing")
    candidates = sorted(inventory - set(VOWELS))
    pin(files, split_path)
    pin(files, cv / "selection.json")
    pin(files, ROOT / config["feature_statistics"])
    run.mkdir(parents=True)
    write_json(
        run / "inventory.json",
        {
            "phones": ordered(inventory),
            "candidates": candidates,
            "scope": "complete_raw_G2P_speech_inventory_of_both_corpora; no_whitelist",
            "fixed_vowels": list(VOWELS),
            "raw_label_mapping": {
                p: canonical(p) for p in ("A", "I", "U", "E", "O", "pau")
            },
            "alignment_aliases_not_merged": {"v": "b", "ty": "ch", "cl": "q"},
            "note": "v and ty retain the original G2P labels although their aligner acoustic models use b/ch",
        },
    )
    corpus_counts, support = Counter(), defaultdict(Counter)
    probes, samples = [], {}
    train_index = 0
    with (
        (run / "train-features.f32").open("xb") as cache,
        (run / "train-segments.jsonl").open("x") as accepted,
        (run / "train-excluded.jsonl").open("x") as excluded,
    ):
        for number, source in enumerate(
            sorted(s for s, m in metadata.items() if m["split"] == "train"), 1
        ):
            meta = metadata[source]
            if not alignment_paths[source].exists():
                continue
            if (
                meta["speaker_id"] not in train_speakers
                or meta["evaluation_role"] != "training"
            ):
                raise ValueError("non-training source in train pool")
            qc = Counter()
            for row, clip in extract(alignment_paths[source], meta, config, files, qc):
                row["corpus"] = meta["corpus"]
                if row["quality_flags"]:
                    excluded.write(json.dumps(row, allow_nan=False) + "\n")
                    continue
                phone = row["phoneme"]
                row["cache_index"] = train_index
                train_index += 1
                cache.write(
                    pool_features(pipe, clip, row["segment_id"]).astype("<f4").tobytes()
                )
                accepted.write(json.dumps(row, allow_nan=False) + "\n")
                support[phone][row["speaker_id"]] += 1
                corpus_counts[f"{meta['corpus']}/{phone}"] += 1
                key = (meta["corpus"], phone)
                if key not in samples or rank(
                    config["draw_seed"], row["segment_id"]
                ) < rank(config["draw_seed"], samples[key][0]["segment_id"]):
                    samples[key] = (dict(row), meta)
                if phone not in {p[0] for p in probes}:
                    probes.append((phone, row["segment_id"], clip.copy()))
            corpus_counts.update({f"{meta['corpus']}/qc/{k}": v for k, v in qc.items()})
            if number % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "train_cache",
                            "utterances": number,
                            "intervals": train_index,
                        }
                    ),
                    flush=True,
                )
    phone_support = {
        p: {
            "eligible_intervals": sum(support[p].values()),
            "speakers_with_two_intervals": sum(n >= 2 for n in support[p].values()),
            "speakers_with_any_interval": len(support[p]),
            "eligible_by_speaker": dict(support[p]),
            "trainable": any(n >= 2 for n in support[p].values()),
            "status": "trainable"
            if any(n >= 2 for n in support[p].values())
            else "awaiting_training_data; remains_in_candidate_box",
        }
        for p in ordered(inventory)
    }
    if any(phone_support[p]["speakers_with_two_intervals"] != 140 for p in VOWELS):
        raise ValueError("fixed vowel training speaker coverage changed")
    expanded.seed_everything(config["training_seed"])
    error = check_cache_equivalence(
        create_encoder(config["encoder"]), pipe, [(i, c) for _, i, c in probes]
    )
    evaluation_report = {}
    for which in ("validation", "test"):
        original_path = ROOT / config["reference_inputs"] / f"{which}-inputs.json"
        original = read_json(original_path)
        pin(files, original_path)
        speakers = original["speakers"]
        if set(speakers) != set(split[which]) or set(speakers) & set(train_speakers):
            raise ValueError("evaluation speaker leak")
        enrollment_sources = defaultdict(set)
        for entry in original["enrollment"]:
            for row in entry["ranks"]:
                enrollment_sources[row["speaker_id"]].add(row["source_file"])
        enrollment = {
            source for sources in enrollment_sources.values() for source in sources
        }
        query_sources = {q["source_file"] for q in original["queries"]}
        if enrollment & query_sources:
            raise ValueError("enrollment/query source overlap")
        all_rows, by_source = [], {}
        with (run / f"{which}-features.f32").open("xb") as cache:
            for source in sorted(enrollment | query_sources):
                meta = metadata[source]
                if meta["split"] != which or meta["speaker_id"] not in speakers:
                    raise ValueError("evaluation source mismatch")
                eligible = []
                for row, clip in extract(
                    alignment_paths[source], meta, config, files, Counter()
                ):
                    if row["quality_flags"]:
                        continue
                    row["cache_index"] = len(all_rows)
                    cache.write(
                        pool_features(pipe, clip, row["segment_id"])
                        .astype("<f4")
                        .tobytes()
                    )
                    all_rows.append(row)
                    eligible.append(row)
                by_source[source] = eligible
        profiles = {}
        for speaker in speakers:
            pool = [
                r
                for source in sorted(enrollment_sources[speaker])
                for r in by_source[source]
            ]
            profiles[speaker] = {
                p: [
                    r["cache_index"]
                    for r in diverse(
                        [r for r in pool if r["phoneme"] == p],
                        config["enrollment_per_phone_maximum"],
                        config["draw_seed"],
                    )
                ]
                for p in ordered(inventory)
            }
        universally_registered = [
            p for p in ordered(inventory) if all(profiles[s][p] for s in speakers)
        ]
        if not set(VOWELS) <= set(universally_registered):
            raise ValueError("fixed vowels missing in enrollment")
        queries = []
        for query in original["queries"]:
            pool = by_source[query["source_file"]]
            groups = {
                p: [r["cache_index"] for r in pool if r["phoneme"] == p]
                for p in ordered(inventory)
            }
            queries.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "source_file": query["source_file"],
                    "role": query["role"],
                    "groups": groups,
                    "scorable": all(groups[p] for p in VOWELS),
                }
            )
        data = {
            "split": which,
            "speakers": speakers,
            "profiles": profiles,
            "queries": queries,
            "universally_registered": universally_registered,
            "feature_count": len(all_rows),
        }
        write_json(run / f"{which}-inputs.json", data)
        write_rows(run / f"{which}-segments.jsonl", all_rows)
        evaluation_report[which] = {
            "feature_count": len(all_rows),
            "registered_for_all_speakers": universally_registered,
            "query_counts": dict(Counter(q["role"] for q in queries)),
            "scored_query_counts": dict(
                Counter(q["role"] for q in queries if q["scorable"])
            ),
        }
        print(
            json.dumps(
                {
                    "stage": "evaluation_cache",
                    "split": which,
                    **evaluation_report[which],
                }
            ),
            flush=True,
        )
    listening = []
    listening_meta = {}
    for (corpus, phone), (row, meta) in sorted(samples.items()):
        if phone in VOWELS:
            continue
        row.update(
            {
                "item_id": row["segment_id"],
                "first_frame": row["start_frame"],
                "last_frame": row["end_frame"],
                "start_sec": row["start_frame"] / 24000,
                "end_sec": row["end_frame"] / 24000,
                "transcript": meta["transcript"],
            }
        )
        listening.append(row)
        listening_meta[row["utterance_id"]] = meta
    review_shared.stage_audio(listening, listening_meta, run / "listening/media")
    dataset = review_shared.review_dataset(
        listening,
        config["draw_seed"],
        dataset_id="all-phoneme-greedy-listening",
        corpus_title="JVS・Common Voice 全候補音素",
    )
    dataset["title"] = f"全候補音素の試聴（保存なし・{len(listening)}区間）"
    write_json(run / "listening/review-dataset.json", dataset)
    write_rows(run / "listening/sample.jsonl", listening)
    report = {
        "status": "prepared_before_training",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "training_speakers": train_speakers,
        "train_feature_count": train_index,
        "phone_support": phone_support,
        "corpus_counts": dict(corpus_counts),
        "evaluation": evaluation_report,
        "cache_equivalence_max_abs_error": error,
        "test_inference_used": False,
        "human_review_answers_saved": False,
        "formal_review_requested": False,
        "unavailable_alignment_files": missing_alignments,
    }
    write_json(run / "preparation-report.json", report)
    for path in run.iterdir():
        if path.is_file():
            pin(files, path)
    for path in (run / "listening").rglob("*"):
        if path.is_file():
            pin(files, path)
    for path in [CONFIG, *BASE.glob("*.py"), *BASE.glob("tests/*.py"), *PHASE3_FILES()]:
        pin(files, path)
    write_json(
        run / "design-freeze.json",
        {
            "status": "frozen_before_training_and_selection",
            "config": config,
            "files": files,
            "runtime": expanded.runtime(),
            "test_inference_used": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "prepared",
                "candidate_count": len(candidates),
                "train_intervals": train_index,
                "unavailable": [
                    p for p in candidates if not phone_support[p]["trainable"]
                ],
                "cache_error": error,
            }
        ),
        flush=True,
    )


def PHASE3_FILES():
    from common import PHASE3

    return [
        *(PHASE3 / "phase3_train").glob("*.py"),
        *(PHASE3 / "phase3_data").glob("*.py"),
        *(PHASE3 / "config").glob("*.json"),
        BASE.parent / "phoneme-training-data-expansion/training.py",
        BASE.parent / "phoneme-consonant-review/sample_review.py",
    ]


if __name__ == "__main__":
    prepare()
