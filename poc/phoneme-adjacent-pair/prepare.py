"""Extract continuous C+V audio while reusing exact separate-phone features."""

import wave
from collections import Counter, defaultdict

from inventory import collect, inventory, layout
from pair_common import (
    BASE,
    CONFIG,
    ROOT,
    check,
    features,
    np,
    pin,
    pipeline,
    pooled,
    rank,
    read,
    sha,
    sources,
    state,
    torch,
    write,
)


def diverse(units, ids, maximum, seed):
    groups = defaultdict(list)
    for i in sorted(ids, key=lambda i: (rank(seed, units[i]["unit_id"]), i)):
        groups[units[i]["source_file"]].append(i)
    ordered = sorted(groups, key=lambda s: (rank(seed, s), s))
    result = []
    while ordered and len(result) < maximum:
        for source in ordered.copy():
            result.append(groups[source].pop(0))
            if not groups[source]:
                ordered.remove(source)
            if len(result) == maximum:
                break
    return result


def evaluation_layout(units, original, config, selected):
    by_source = defaultdict(list)
    for i, u in enumerate(units):
        by_source[u["source_file"]].append(i)
    profiles = {}
    for speaker in original["speakers"]:
        ids = [
            i
            for source in original["enrollment_sources"][speaker]
            for i in by_source[source]
        ]
        assert all(units[i]["speaker_id"] == speaker for i in ids)
        profiles[speaker] = {
            p: diverse(
                units,
                [i for i in ids if units[i]["pair"] == p],
                config["enrollment_limit_per_pair"],
                config["selection_seed"],
            )
            for p in selected
        }
    universal = [p for p in selected if all(profiles[s][p] for s in profiles)]
    queries = []
    for query in original["queries"]:
        ids = by_source[query["source_file"]]
        assert all(units[i]["speaker_id"] == query["speaker_id"] for i in ids)
        groups = {p: [i for i in ids if units[i]["pair"] == p] for p in selected}
        queries.append(
            {
                **query,
                "groups": groups,
                "scorable": bool(
                    query["old_eligible"] and any(groups[p] for p in universal)
                ),
            }
        )
    enrollment_ids = {
        i for profile in profiles.values() for ids in profile.values() for i in ids
    }
    query_ids = {i for q in queries for ids in q["groups"].values() for i in ids}
    assert not enrollment_ids & query_ids
    return {
        "speakers": original["speakers"],
        "profiles": profiles,
        "queries": queries,
        "universally_registered": universal,
        "original_layout": original,
    }


def geometry_from_raw(raw, unit):
    i = unit["interval_index"]
    left, right = raw["intervals"][i : i + 2]
    assert raw["source_file"] == unit["source_file"]
    assert raw["source_sha256"] == unit["source_sha256"]
    assert raw["speaker_id"] == unit["speaker_id"]
    labels = [
        p.lower() if p in ("A", "I", "U", "E", "O") else p
        for p in raw["raw_phonemes"][i - 1 : i + 1]
    ]
    assert "+".join(labels) == unit["pair"]
    a, b, c = unit["frames"]
    assert [
        round(left["start_sec"] * 24000),
        round(left["end_sec"] * 24000),
        round(right["start_sec"] * 24000),
        round(right["end_sec"] * 24000),
    ] == [a, b, b, c]


def prepare_dataset(config, run, split, corpus, selected, spec, catalog):
    directory = run / "data" / f"{corpus}-{split}"
    if (directory / "complete.json").exists():
        receipt = read(directory / "complete.json")
        check(receipt["files"])
        assert receipt["selection_sha256"] == sha(run / "pair-selection.json")
        return receipt
    directory.mkdir(parents=True, exist_ok=True)
    units = [u for u in catalog["units"] if u["pair"] in selected]
    matrix = features(spec)
    pipe, statistics_path = pipeline(config)
    files = pin([spec["manifest"], spec["features"], statistics_path])
    if spec["inputs"]:
        files.update(pin([spec["inputs"]]))
    by_source = defaultdict(list)
    for i, u in enumerate(units):
        by_source[u["source_file"]].append(i)
    temporary = directory / "features.tmp.npy"
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="<f4", shape=(len(units), 3, 128)
    )
    probes, maximum_error, frames = [], 0.0, 0
    for number, (source, ids) in enumerate(by_source.items(), 1):
        first = units[ids[0]]
        audio_path, alignment_path = ROOT / source, ROOT / first["alignment_file"]
        assert sha(audio_path) == first["source_sha256"]
        files.update(pin([audio_path, alignment_path]))
        raw = read(alignment_path)
        with wave.open(str(audio_path)) as wav:
            assert (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getcomptype(),
            ) == (24000, 1, 2, "NONE")
            pcm = (
                np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
        for i in ids:
            u = units[i]
            assert (
                u["source_sha256"] == first["source_sha256"]
                and u["alignment_file"] == first["alignment_file"]
            )
            geometry_from_raw(raw, u)
            a, b, c = u["frames"]
            assert 0 <= a < b < c <= len(pcm)
            output[i, :2] = matrix[u["indices"]]
            output[i, 2] = pooled(pipe, pcm[a:c], u["unit_id"])
            # Deterministic stratified probes, independent of model/EER.
            if len(probes) < 100 and (number == 1 or len(probes) < number):
                separate = np.stack(
                    [
                        pooled(pipe, pcm[a:b], u["unit_id"] + "-c"),
                        pooled(pipe, pcm[b:c], u["unit_id"] + "-v"),
                    ]
                )
                error = float(np.max(np.abs(separate - output[i, :2])))
                assert error < 2e-6, error
                maximum_error = max(maximum_error, error)
                probes.append(i)
            frames += c - a
        if number % 1000 == 0:
            output.flush()
            state(
                run,
                "extracting",
                split=split,
                corpus=corpus,
                sources=number,
                total=len(by_source),
            )
    output.flush()
    assert np.isfinite(output).all()
    del output
    temporary.replace(directory / "features.npy")
    write(directory / "units.json", units, immutable=True)
    if catalog["layout"] is not None:
        data = evaluation_layout(units, catalog["layout"], config, selected)
        write(directory / "inputs.json", data, immutable=True)
        files.update(pin([directory / "inputs.json"]))
    files.update(pin([directory / "units.json", directory / "features.npy"]))
    receipt = {
        "files": files,
        "selection_sha256": sha(run / "pair-selection.json"),
        "units": len(units),
        "sources": len(by_source),
        "pair_counts": dict(Counter(u["pair"] for u in units)),
        "unique_frames_split": frames,
        "unique_frames_joined": frames,
        "feature_probes": probes,
        "maximum_separate_feature_difference": maximum_error,
        "feature_shape": [len(units), 3, 128],
    }
    write(directory / "complete.json", receipt, immutable=True)
    return receipt


def prepare(config, run, test=False):
    selection = inventory(config, run)
    selected = selection["selected"]
    assert selected, "No pair passed prespecified support requirements"
    source = ROOT / config["source_run"]
    phones = read(source / "design-freeze.json")["phones"]
    if test:
        # All trained models/validation thresholds precede test feature extraction.
        locked = read(run / "thresholds-freeze.json")
        check(locked["files"])
    for split in ["test"] if test else ["train", "validation"]:
        for c, spec in sources(config, split).items():
            if test:
                units, lookup, _ = collect(spec, config, phones)
                catalog = {
                    "units": units,
                    "layout": layout(read(spec["inputs"]), lookup),
                }
            else:
                catalog = read(ROOT / selection["catalogs"][f"{c}-{split}"])
            prepare_dataset(config, run, split, c, selected, spec, catalog)
    paths = [
        run / "pair-selection.json",
        BASE / "pair_common.py",
        BASE / "inventory.py",
        BASE / "prepare.py",
    ]
    paths += list((BASE.parent / "phoneme-speaker-encoder/phase3_data").glob("*.py"))
    paths += list((BASE.parent / "phoneme-speaker-encoder/phase3_train").glob("*.py"))
    paths += list((BASE.parent / "phoneme-speaker-encoder/config").glob("*.json"))
    files = pin(paths)
    for path in (run / "data").glob(
        f"*-{'test' if test else 'validation'}/complete.json"
    ):
        files.update(read(path)["files"])
        files.update(pin([path]))
    if not test:
        path = run / "data/train-train/complete.json"
        files.update(read(path)["files"])
        files.update(pin([path]))
    write(
        run / ("test-prepared.json" if test else "prepared-freeze.json"),
        {"files": files, "selected": selected},
        immutable=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    config = read(CONFIG)
    prepare(config, ROOT / config["run_directory"])
