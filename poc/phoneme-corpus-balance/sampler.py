"""Capacity-constrained corpus quotas, preserving the original phone/pair budget."""

import numpy as np

CORPORA = ("jvs", "common_voice", "jtube")


def corpus(speaker):
    return (
        "jtube"
        if speaker.startswith("jtube_")
        else "common_voice"
        if speaker.startswith("cv17_")
        else "jvs"
    )


def quotas(capacities, cycle, groups):
    assert cycle and set(cycle) == set(CORPORA)
    assert all(capacities[c] >= 0 for c in CORPORA)
    width = min(10, sum(capacities.values()))
    if width == 0:
        raise ValueError("no positive pair for this phone")
    counts = np.zeros((groups, 3), dtype=np.int64)
    cursor = 0
    for row in range(groups):
        for _ in range(width):
            while True:
                name = cycle[cursor % len(cycle)]
                cursor += 1
                column = CORPORA.index(name)
                if counts[row, column] < capacities[name]:
                    counts[row, column] += 1
                    break
    return counts


def schedule(groups, phones, updates, seed, speaker_to_class, *, cycle, rank):
    matrix = np.full((updates * 5, 20), -1, dtype=np.int64)
    labels = np.zeros_like(matrix)
    valid = np.zeros_like(matrix, dtype=bool)
    for position, phone in enumerate(phones):
        eligible = {
            s: np.asarray(ids, dtype=np.int64)
            for (s, p), ids in groups.items()
            if p == phone and len(ids) >= 2
        }
        by_corpus = {
            c: sorted(
                (s for s in eligible if corpus(s) == c),
                key=lambda s: (rank(seed, f"speaker/{phone}/{s}"), s),
            )
            for c in CORPORA
        }
        slots = np.arange(position, updates * 5, len(phones))
        counts = quotas({c: len(v) for c, v in by_corpus.items()}, cycle, len(slots))
        for column, name in enumerate(CORPORA):
            speakers = by_corpus[name]
            if not speakers:
                continue
            widths = counts[:, column]
            prior = np.cumsum(widths) - widths
            col = np.arange(10)[None, :]
            used = col < widths[:, None]
            selected = (prior[:, None] + col) % len(speakers)
            offsets = counts[:, :column].sum(axis=1)
            for i, speaker in enumerate(speakers):
                rr, cc = np.where(used & (selected == i))
                generator = np.random.default_rng(
                    int(rank(seed, f"segments/{phone}/{speaker}")[:16], 16)
                )
                ids = eligible[speaker]
                first = generator.integers(len(ids), size=len(rr))
                second = generator.integers(len(ids) - 1, size=len(rr))
                second += second >= first
                for offset, choices in ((0, first), (1, second)):
                    target = (offsets[rr] + cc) * 2 + offset
                    matrix[slots[rr], target] = ids[choices]
                    labels[slots[rr], target] = speaker_to_class[speaker]
                    valid[slots[rr], target] = True
    assert valid.any(axis=1).all() and not np.any(valid.sum(axis=1) % 2)
    return (
        matrix.reshape(updates, 100),
        labels.reshape(updates, 100),
        valid.reshape(updates, 100),
    )


def exposure(labels, valid, phones, speakers):
    """Measure actual valid examples, including rare-phone redistribution."""
    labels, valid = labels.reshape(-1, 20), valid.reshape(-1, 20)
    code = np.array([CORPORA.index(corpus(s)) for s in speakers])
    total = np.bincount(code[labels[valid]], minlength=3)
    by_phone = {}
    for i, phone in enumerate(phones):
        values = labels[i :: len(phones)]
        mask = valid[i :: len(phones)]
        count = np.bincount(code[values[mask]], minlength=3)
        by_phone[phone] = dict(zip(CORPORA, count.tolist(), strict=True))
    return {
        "valid_examples": int(total.sum()),
        "corpus_examples": dict(zip(CORPORA, total.tolist(), strict=True)),
        "corpus_fractions": dict(
            zip(CORPORA, (total / total.sum()).tolist(), strict=True)
        ),
        "phone_corpus_examples": by_phone,
    }
