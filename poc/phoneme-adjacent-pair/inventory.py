"""Count genuine adjacent tokens before selecting types; never use test counts."""

from collections import Counter, defaultdict

from pair_common import (
    CONFIG,
    ROOT,
    VOWELS,
    check,
    corpus,
    pin,
    read,
    rows,
    sources,
    state,
    write,
)


def candidate(left, right, config, consonants):
    if left is None or left["source_file"] != right["source_file"]:
        return None
    if left["phoneme"] not in consonants or right["phoneme"] not in VOWELS:
        return None
    if right["interval_index"] != left["interval_index"] + 1:
        return None
    a, b, b2, c = (
        left["raw_start_frame"],
        left["raw_end_frame"],
        right["raw_start_frame"],
        right["raw_end_frame"],
    )
    if (
        b != b2
        or min(b - a, c - b) < config["minimum_phone_frames"]
        or c - a > config["maximum_pair_frames"]
    ):
        return None
    assert left["speaker_id"] == right["speaker_id"]
    assert left["source_sha256"] == right["source_sha256"]
    assert left["alignment_file"] == right["alignment_file"]
    assert (
        left["start_frame"],
        left["end_frame"],
        right["start_frame"],
        right["end_frame"],
    ) == (a, b, b, c)
    assert not left["quality_flags"] and not right["quality_flags"]
    return {
        "unit_id": left["segment_id"] + "__" + right["segment_id"],
        "pair": left["phoneme"] + "+" + right["phoneme"],
        "speaker_id": left["speaker_id"],
        "source_file": left["source_file"],
        "source_sha256": left["source_sha256"],
        "alignment_file": left["alignment_file"],
        "interval_index": left["interval_index"],
        "frames": [a, b, c],
        "indices": [left["cache_index"], right["cache_index"]],
    }


def collect(spec, config, phones):
    consonants = set(phones) - VOWELS - {"N", "cl"}
    units, lookup, source_meta = [], {}, {}
    previous = None
    for row in rows(spec["manifest"]):
        lookup[row["cache_index"]] = row["source_file"]
        source_meta[row["source_file"]] = (row["speaker_id"], row["source_sha256"])
        found = candidate(previous, row, config, consonants)
        if found:
            units.append(found)
        previous = row
    return units, lookup, source_meta


def layout(data, lookup):
    enrollment = {
        s: sorted({lookup[i] for ids in profile.values() for i in ids})
        for s, profile in data["profiles"].items()
    }
    queries = []
    for row in data["queries"]:
        inferred = {lookup[i] for ids in row["groups"].values() for i in ids}
        assert len(inferred) <= 1
        source = row.get("source_file", next(iter(inferred), None))
        if inferred:
            assert source in inferred
        queries.append(
            {
                "query_id": row["query_id"],
                "speaker_id": row["speaker_id"],
                "role": row["role"],
                "source_file": source,
                "old_eligible": all(row["groups"].get(v) for v in VOWELS)
                and VOWELS <= set(data["universally_registered"]),
            }
        )
    assert not {s for values in enrollment.values() for s in values} & {
        q["source_file"] for q in queries
    }
    return {
        "speakers": data["speakers"],
        "enrollment_sources": enrollment,
        "queries": queries,
    }


def inventory(config, run):
    if (run / "pair-selection.json").exists():
        result = read(run / "pair-selection.json")
        assert result["config"] == config
        check(result["files"])
        return result
    run.mkdir(parents=True, exist_ok=True)
    source = ROOT / config["source_run"]
    phones = read(source / "design-freeze.json")["phones"]
    files = pin(
        [CONFIG, source / "design-freeze.json", source / "preparation-report.json"]
    )
    tables, catalogs = {}, {}
    train_groups, train_speakers = Counter(), defaultdict(set)
    for split in ("train", "validation"):
        for c, spec in sources(config, split).items():
            state(run, "inventory", split=split, corpus=c)
            units, lookup, metadata = collect(spec, config, phones)
            files.update(
                pin(
                    [
                        spec["manifest"],
                        spec["features"],
                        *([spec["inputs"]] if spec["inputs"] else []),
                    ]
                )
            )
            if split == "train":
                for speaker, _ in metadata.values():
                    train_speakers[corpus(speaker)].add(speaker)
                train_groups.update((u["pair"], u["speaker_id"]) for u in units)
                info = None
            else:
                info = layout(read(spec["inputs"]), lookup)
                by_source = defaultdict(set)
                enrollment_count = Counter()
                for u in units:
                    by_source[u["source_file"]].add(u["pair"])
                    if u["source_file"] in set(
                        info["enrollment_sources"][u["speaker_id"]]
                    ):
                        enrollment_count[u["pair"], u["speaker_id"]] += 1
                normal = [q for q in info["queries"] if q["role"] == "verification"]
                types = sorted({u["pair"] for u in units})
                tables[c] = {
                    p: {
                        "enrollment_speakers": sum(
                            enrollment_count[p, s] > 0 for s in info["speakers"]
                        ),
                        "total_speakers": len(info["speakers"]),
                        "query_count": sum(
                            q["old_eligible"] and p in by_source[q["source_file"]]
                            for q in normal
                        ),
                        "query_total": len(normal),
                    }
                    for p in types
                }
            target = run / "inventory" / f"{c}-{split}.json"
            write(target, {"units": units, "layout": info}, immutable=True)
            catalogs[f"{c}-{split}"] = str(target.relative_to(ROOT))
            files.update(pin([target]))
    types = sorted({p for p, s in train_groups})
    support, eligible = {}, []
    for p in types:
        train = {
            c: {
                "speakers_with_two": sum(train_groups[p, s] >= 2 for s in ss),
                "total_speakers": len(ss),
                "tokens": sum(train_groups[p, s] for s in ss),
            }
            for c, ss in train_speakers.items()
        }
        validation = {
            c: tables[c].get(
                p,
                {
                    "enrollment_speakers": 0,
                    "total_speakers": 1,
                    "query_count": 0,
                    "query_total": 1,
                },
            )
            for c in config["corpora"]
        }
        coverage = min(v["query_count"] / v["query_total"] for v in validation.values())
        valid = (
            all(
                v["speakers_with_two"] / v["total_speakers"]
                >= config["minimum_train_speaker_fraction_per_corpus"]
                for v in train.values()
            )
            and all(
                v["enrollment_speakers"] == v["total_speakers"]
                for v in validation.values()
            )
            and coverage >= config["minimum_validation_query_fraction"]
        )
        support[p] = {
            "train": train,
            "validation": validation,
            "minimum_query_fraction": coverage,
            "eligible": valid,
        }
        if valid:
            eligible.append(p)
    selected = sorted(
        eligible, key=lambda p: (-support[p]["minimum_query_fraction"], p)
    )[: config["maximum_pair_types"]]
    result = {
        "config": config,
        "selected": sorted(selected),
        "eligible_ranked": sorted(
            eligible, key=lambda p: (-support[p]["minimum_query_fraction"], p)
        ),
        "support": support,
        "catalogs": catalogs,
        "files": files,
        "test_read_for_selection": False,
    }
    write(run / "pair-selection.json", result, immutable=True)
    print(
        {
            "selected": result["selected"],
            "eligible": len(eligible),
            "candidates": len(types),
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    config = read(CONFIG)
    inventory(config, ROOT / config["run_directory"])
