"""Disjoint WAV enrollment/query inputs for fusion on a frozen 280-label encoder."""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-corpus-seed-replication"))
import replicate as r

s = r.s
BACKEND = BASE.parent / "phoneme-transformer-fusion/backend.py"
spec = importlib.util.spec_from_file_location("frozen_fusion_backend", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def corpus(speaker):
    return (
        "jtube"
        if speaker.startswith("jtube_")
        else "common_voice"
        if speaker.startswith("cv17_")
        else "jvs"
    )


def select_sources(catalog, cap, seed):
    enroll, query, excluded = set(), set(), {}
    for speaker, sources in sorted(catalog.items()):
        ordered = sorted(sources, key=lambda digest: (s.rank(seed, digest), digest))
        n = min(cap, len(ordered) // 2)
        if n == 0:
            excluded[speaker] = "fewer_than_two_distinct_wavs"
        else:
            enroll.update(ordered[:n])
            query.update(ordered[n:])
    assert not enroll & query
    return enroll, query, excluded


def diverse(pool, limit, seed):
    groups = defaultdict(list)
    for index, digest, identifier in pool:
        groups[digest].append((s.rank(seed, identifier), index))
    ordered = [
        sorted(groups[digest])
        for digest in sorted(groups, key=lambda v: (s.rank(seed, v), v))
    ]
    result = []
    depth = 0
    while len(result) < limit:
        available = [group[depth][1] for group in ordered if depth < len(group)]
        if not available:
            break
        result.extend(available[: limit - len(result)])
        depth += 1
    return result


def training_inputs(config, manifest, phones, evaluation_speakers, evaluation_hashes):
    catalog, owner = defaultdict(set), {}
    for row in s.rows(manifest):
        assert row["split"] == "train" and not row["quality_flags"]
        speaker, digest = row["speaker_id"], row["source_sha256"]
        assert speaker not in evaluation_speakers and digest not in evaluation_hashes
        assert owner.setdefault(digest, speaker) == speaker, (
            "WAV belongs to multiple labels"
        )
        catalog[speaker].add(digest)
    enroll, query, excluded = select_sources(
        catalog, config["training_enrollment_wavs"], config["data_seed"]
    )
    pools, queries, metadata = defaultdict(list), {}, []
    for index, row in enumerate(s.rows(manifest)):
        assert row["cache_index"] == index
        metadata.append(
            [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
        )
        speaker, digest, phone = row["speaker_id"], row["source_sha256"], row["phoneme"]
        if digest in enroll:
            pools[speaker, phone].append((index, digest, row["segment_id"]))
        elif digest in query:
            q = queries.setdefault(
                digest,
                {
                    "query_id": digest,
                    "speaker_id": speaker,
                    "corpus": corpus(speaker),
                    "groups": {p: [] for p in phones},
                },
            )
            q["groups"][phone].append(index)
    profiles = {
        speaker: {
            p: diverse(
                pools[speaker, p], config["enrollment_limit"], config["data_seed"]
            )
            for p in phones
        }
        for speaker in catalog
        if speaker not in excluded
    }
    eligible = [
        q for _, q in sorted(queries.items()) if all(q["groups"][p] for p in s.VOWELS)
    ]
    query_speakers = {q["speaker_id"] for q in eligible}
    for speaker, profile in profiles.items():
        if not all(profile[p] for p in s.VOWELS):
            excluded[speaker] = "enrollment_missing_vowels"
        elif speaker not in query_speakers:
            excluded[speaker] = "no_query_with_five_vowels"
    speakers = sorted(set(catalog) - set(excluded))
    assert all(sum(corpus(v) == c for v in speakers) >= 2 for c in s.CORPORA)
    return {
        "speakers": speakers,
        "profiles": {v: profiles[v] for v in speakers},
        "queries": [q for q in eligible if q["speaker_id"] in speakers],
        "excluded_labels": excluded,
        "encoder_training_labels": len(catalog),
        "training_label_counts": {
            c: sum(corpus(v) == c for v in speakers) for c in s.CORPORA
        },
        "enrollment_wav_hashes": sorted(enroll),
        "query_wav_hashes": sorted(query),
        "queries_missing_vowels": len(queries) - len(eligible),
        "feature_count": len(metadata),
    }, s.np.asarray(metadata, dtype=s.np.float32)


def aggregate(indices, vectors, metadata, normalize):
    if not indices:
        return s.np.zeros(128, s.np.float32), s.np.zeros(4, s.np.float32), False
    mean = vectors[indices].mean(axis=0)
    norm = s.np.linalg.norm(mean)
    assert norm > 1e-12
    info = s.np.array(
        [
            s.np.log1p(len(indices)),
            s.np.log(metadata[indices, 0].mean()),
            1 - norm,
            metadata[indices, 1].mean() / 50,
        ],
        dtype=s.np.float32,
    )
    return mean / norm if normalize else mean, info, True


def aggregates(data, vectors, metadata, phones):
    result = {}
    for name, groups, normalize in (
        ("enrollment", [data["profiles"][v] for v in data["speakers"]], True),
        ("queries", [q["groups"] for q in data["queries"]], False),
    ):
        values = s.np.zeros((len(groups), len(phones), 128), s.np.float32)
        info = s.np.zeros((len(groups), len(phones), 4), s.np.float32)
        mask = s.np.zeros((len(groups), len(phones)), bool)
        for i, group in enumerate(groups):
            for j, p in enumerate(phones):
                values[i, j], info[i, j], mask[i, j] = aggregate(
                    group.get(p, []), vectors, metadata, normalize
                )
        result[name] = values
        prefix = "e" if name == "enrollment" else "q"
        result[prefix + "meta"], result[prefix + "mask"] = info, mask
    return result


def make_schedule(config, data, seed):
    rng = s.np.random.default_rng(seed)
    speakers = data["speakers"]
    by_speaker = [
        [i for i, q in enumerate(data["queries"]) if q["speaker_id"] == v]
        for v in speakers
    ]
    negatives = [
        [j for j, v in enumerate(speakers) if j != i and corpus(v) == corpus(speaker)]
        for i, speaker in enumerate(speakers)
    ]
    shape = config["updates"], config["queries_per_update"]
    selected = rng.integers(len(speakers), size=shape)
    qids, claims = (
        s.np.empty(shape, dtype=s.np.int32),
        s.np.empty(shape, dtype=s.np.int16),
    )
    for u in range(shape[0]):
        for j, i in enumerate(selected[u]):
            qids[u, j], claims[u, j] = (
                rng.choice(by_speaker[i]),
                rng.choice(negatives[i]),
            )
    return {"queries": qids, "negative_claims": claims}


def batch(arrays, query_ids, claims, mask=None):
    f, c = backend.pair_features(
        arrays["enrollment"][claims],
        arrays["emeta"][claims],
        arrays["queries"][query_ids],
        arrays["qmeta"][query_ids],
    )
    available = arrays["emask"][claims] & arrays["qmask"][query_ids]
    return f, c, available if mask is None else available & mask


def training_batch(arrays, data, schedule, update):
    q = s.torch.from_numpy(schedule["queries"][update].astype(s.np.int64))
    negative = s.torch.from_numpy(
        schedule["negative_claims"][update].astype(s.np.int64)
    )
    index = {v: i for i, v in enumerate(data["speakers"])}
    genuine = s.torch.tensor([index[data["queries"][int(i)]["speaker_id"]] for i in q])
    shared = arrays["qmask"][q] & arrays["emask"][genuine] & arrays["emask"][negative]
    assert shared[:, :5].all()
    f, c, mask = batch(
        arrays,
        s.torch.cat((q, q)),
        s.torch.cat((genuine, negative)),
        s.torch.cat((shared, shared)),
    )
    return f, c, mask, s.torch.cat((s.torch.ones(len(q)), s.torch.zeros(len(q))))
