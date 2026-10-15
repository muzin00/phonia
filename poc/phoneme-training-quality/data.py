"""Remap both acoustic and Transformer inputs after interval-level filtering."""

from collections import Counter

import numpy as np


def subset_pool(info, arrays, keep):
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != arrays["segment_speaker"].shape or not keep.any():
        raise ValueError("Invalid or empty training selection")
    ids = np.flatnonzero(keep)
    mapping = np.full(len(keep), -1, np.int64)
    mapping[ids] = np.arange(len(ids))
    speakers_kept = np.unique(arrays["segment_speaker"][ids])
    speaker_map = np.full(len(info["speakers"]), -1, np.int64)
    speaker_map[speakers_kept] = np.arange(len(speakers_kept))
    groups, owners, phones = [], [], []
    for i, (a, b) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:])):
        group = mapping[arrays["indices"][a:b]]
        group = group[group >= 0]
        if len(group):
            groups.append(group)
            owners.append(speaker_map[arrays["group_speaker"][i]])
            phones.append(arrays["group_phone"][i])
    result = {
        "indices": np.concatenate(groups),
        "offsets": np.r_[0, np.cumsum([len(g) for g in groups])],
        "group_speaker": np.asarray(owners),
        "group_phone": np.asarray(phones),
        "segment_speaker": speaker_map[arrays["segment_speaker"][ids]].astype(np.int16),
        **{k: arrays[k][ids] for k in ("segment_phone", "recording", "metadata")},
    }
    if not np.array_equal(np.sort(result["indices"]), np.arange(len(ids))):
        raise ValueError(
            "Training groups must cover each retained interval exactly once"
        )
    stats = {
        "phones": info["phones"],
        "speakers": [info["speakers"][i] for i in speakers_kept],
        "segments": len(ids),
        "pair_eligible_segments": sum(len(g) for g in groups if len(g) >= 2),
        "singleton_segments": sum(len(g) == 1 for g in groups),
        "speaker_phone_groups": len(groups),
        "phones_without_positive_pair": [
            p
            for i, p in enumerate(info["phones"])
            if not any(len(g) >= 2 for g, phone in zip(groups, phones) if phone == i)
        ],
    }
    return stats, result, ids, mapping


def subset_bank(bank, mapping, vowels):
    dropped = Counter()

    def remap(groups):
        output = {}
        for p, ids in groups.items():
            ids = np.asarray(ids, dtype=np.int64)
            if np.any((ids < 0) | (ids >= len(mapping))):
                raise ValueError("Source bank index outside original pool")
            values = mapping[ids]
            output[p] = values[values >= 0].tolist()
        return output

    profiles = {s: remap(bank["profiles"][s]) for s in bank["speakers"]}
    eligible = {s for s, g in profiles.items() if all(g.get(p) for p in vowels)}
    dropped["profiles_missing_vowels"] = len(profiles) - len(eligible)
    queries = []
    for q in bank["queries"]:
        if q["speaker_id"] not in eligible:
            dropped["queries_ineligible_profile"] += 1
            continue
        groups = remap(q["groups"])
        if not all(groups.get(p) for p in vowels):
            dropped["queries_missing_vowels"] += 1
            continue
        queries.append({**q, "groups": groups})
    owners = {q["speaker_id"] for q in queries}
    dropped["profiles_without_queries"] = len(eligible - owners)
    speakers = [s for s in bank["speakers"] if s in eligible & owners]
    return {
        "speakers": speakers,
        "profiles": {s: profiles[s] for s in speakers},
        "queries": queries,
        "feature_count": int(np.count_nonzero(mapping >= 0)),
    }, dict(dropped)


def validate_bank(bank, info, arrays):
    speakers = {s: i for i, s in enumerate(info["speakers"])}
    phones = {p: i for i, p in enumerate(info["phones"])}
    profile_recordings = {}
    items = [(s, g, True) for s, g in bank["profiles"].items()]
    items += [(q["speaker_id"], q["groups"], False) for q in bank["queries"]]
    for speaker, groups, profile in items:
        if speaker not in bank["speakers"] or not all(
            groups.get(p) for p in info["phones"][:5]
        ):
            raise ValueError("Training five-vowel/owner gate failed")
        recordings = set()
        for phone, values in groups.items():
            ids = np.asarray(values, np.int64)
            if np.any((ids < 0) | (ids >= info["segments"])):
                raise ValueError("Remapped training index out of bounds")
            if not np.all(arrays["segment_speaker"][ids] == speakers[speaker]):
                raise ValueError("Remapped speaker mismatch")
            if not np.all(arrays["segment_phone"][ids] == phones[phone]):
                raise ValueError("Remapped phoneme mismatch")
            recordings.update(arrays["recording"][ids].tolist())
        if profile:
            profile_recordings[speaker] = recordings
        elif recordings & profile_recordings[speaker]:
            raise ValueError("Enrollment/query recording overlap")
    if set(bank["speakers"]) != {q["speaker_id"] for q in bank["queries"]}:
        raise ValueError("Training speaker has no query")
    return {
        "speakers": len(bank["speakers"]),
        "queries": len(bank["queries"]),
        "segment_ownership_and_recording_separation_verified": True,
    }
