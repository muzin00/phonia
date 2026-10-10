"""Eight-phone sampling with the existing 100-example optimizer budget."""

from __future__ import annotations

import random

from phase3_data.manifest import Segment, json_sha256, seeded_hash
from phase3_data.sampling import BalancedSampler, SampleRequest

PHONEMES = ("a", "i", "u", "e", "o", "m", "n", "s")


class PhonemeSampler(BalancedSampler):
    """Keep ten speakers, five complete phone groups and two positives each.

    Every eight updates expose each phone five times. The inherited checkpoint
    replay balances speaker selections independently of DataLoader prefetch.
    ``Segment.vowel`` is the legacy field name for the phoneme label.
    """

    def __init__(self, segments: list[Segment], run_seed: int):
        super().__init__(segments, run_seed)
        for speaker in self.speakers:
            for phone in PHONEMES:
                ids = self.groups.get((speaker, phone), [])
                if len(ids) < 2 or len(set(ids)) != len(ids):
                    raise ValueError(f"need two distinct {speaker}/{phone} segments")
                self.groups[(speaker, phone)] = sorted(ids)
        if any(s.vowel not in PHONEMES for s in segments):
            raise ValueError("unexpected training phoneme")
        self.cohort_sha256 = json_sha256(
            {
                "policy": "eight_phone_five_group_rotation_random_sample_v1",
                "segments": sorted(
                    (s.speaker_id, s.vowel, s.segment_id) for s in segments
                ),
            }
        )

    @staticmethod
    def phones_at(update: int) -> tuple[str, ...]:
        return tuple(PHONEMES[(update * 5 + offset) % 8] for offset in range(5))

    def batch(self, logical_update: int) -> list[SampleRequest]:
        if logical_update != self.next_update:
            raise ValueError("sampler update mismatch")
        selected = sorted(
            self.speakers,
            key=lambda speaker: (
                self.counts[speaker],
                seeded_hash("speaker", self.run_seed, logical_update, speaker),
                speaker.encode("utf-8"),
            ),
        )[:10]
        result = []
        for phone in self.phones_at(logical_update):
            for speaker in sorted(selected):
                generator = random.Random(
                    seeded_hash(
                        "phoneme_segments",
                        self.run_seed,
                        logical_update,
                        speaker,
                        phone,
                    )
                )
                ids = generator.sample(self.groups[(speaker, phone)], 2)
                result.extend(SampleRequest(sid, logical_update) for sid in sorted(ids))
        for speaker in selected:
            self.counts[speaker] += 1
        self.next_update += 1
        return result
