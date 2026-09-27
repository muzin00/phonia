"""Worker-independent balanced logical batches and vowel microbatches."""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import VOWELS, Segment, json_sha256, seeded_hash


@dataclass(frozen=True, slots=True)
class SampleRequest:
    segment_id: str
    logical_update: int


class BalancedSampler:
    """Produce P=10 x five vowels x K=2 IDs in design-specified order.

    Checkpoint with ``state_at(completed_updates)``. This rebuilds the state at
    the optimizer boundary, ignoring any DataLoader batches prefetched ahead.
    """

    def __init__(self, segments: list[Segment], run_seed: int):
        self.run_seed = run_seed
        self.groups: dict[tuple[str, str], list[str]] = {}
        speakers = set()
        for segment in segments:
            if segment.split != "train" or segment.role != "training":
                raise ValueError("sampler accepts only train/training segments")
            speakers.add(segment.speaker_id)
            self.groups.setdefault((segment.speaker_id, segment.vowel), []).append(
                segment.segment_id
            )
        if len(speakers) < 10:
            raise ValueError("balanced sampler requires at least 10 speakers")
        self.speakers = sorted(speakers)
        self.cohort_sha256 = json_sha256(
            sorted((s.speaker_id, s.vowel, s.segment_id) for s in segments)
        )
        for speaker in self.speakers:
            for vowel in VOWELS:
                ids = self.groups.get((speaker, vowel), [])
                if len(ids) < 2 or len(set(ids)) != len(ids):
                    raise ValueError(
                        f"need two distinct {speaker}/{vowel} training segments"
                    )
        self.counts = {speaker: 0 for speaker in self.speakers}
        self.next_update = 0

    def batch(self, logical_update: int) -> list[SampleRequest]:
        if logical_update != self.next_update:
            raise ValueError(
                f"expected logical update {self.next_update}, got {logical_update}"
            )
        selected = sorted(
            self.speakers,
            key=lambda speaker: (
                self.counts[speaker],
                seeded_hash("speaker", self.run_seed, logical_update, speaker),
                speaker.encode("utf-8"),
            ),
        )[:10]
        for speaker in selected:
            self.counts[speaker] += 1
        self.next_update += 1
        result = []
        for vowel in VOWELS:
            for speaker in sorted(selected):
                ids = sorted(
                    self.groups[(speaker, vowel)],
                    key=lambda segment_id: (
                        seeded_hash(
                            "segment",
                            self.run_seed,
                            logical_update,
                            speaker,
                            vowel,
                            segment_id,
                        ),
                        segment_id.encode("utf-8"),
                    ),
                )[:2]
                result.extend(
                    SampleRequest(segment_id, logical_update)
                    for segment_id in sorted(ids)
                )
        return result

    def microbatches(self, logical_update: int) -> list[list[SampleRequest]]:
        batch = self.batch(logical_update)
        return [batch[index : index + 20] for index in range(0, 100, 20)]

    def state_at(self, completed_updates: int) -> dict:
        if completed_updates < 0:
            raise ValueError("completed_updates must be nonnegative")
        if completed_updates == self.next_update:
            counts = self.counts
        else:
            counts = {speaker: 0 for speaker in self.speakers}
            for update in range(completed_updates):
                selected = sorted(
                    self.speakers,
                    key=lambda speaker: (
                        counts[speaker],
                        seeded_hash("speaker", self.run_seed, update, speaker),
                        speaker.encode("utf-8"),
                    ),
                )[:10]
                for speaker in selected:
                    counts[speaker] += 1
        return {
            "run_seed": self.run_seed,
            "next_logical_update": completed_updates,
            "cohort_sha256": self.cohort_sha256,
            "speaker_counts": dict(counts),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("run_seed") != self.run_seed
            or state.get("cohort_sha256") != self.cohort_sha256
            or set(state.get("speaker_counts", {})) != set(self.speakers)
        ):
            raise ValueError("sampler state does not match seed or cohort")
        update = state.get("next_logical_update")
        expected = self.state_at(update)
        if state != expected:
            raise ValueError("sampler state checksum/count mismatch")
        self.counts = dict(state["speaker_counts"])
        self.next_update = update


class VowelMicrobatchSampler:
    """PyTorch BatchSampler: five complete 20-item groups per logical update."""

    def __init__(self, sampler: BalancedSampler, *, start_update: int, updates: int):
        if updates < 0 or start_update < 0:
            raise ValueError("invalid update range")
        self.sampler, self.start_update, self.updates = sampler, start_update, updates

    def __len__(self) -> int:
        return 5 * self.updates

    def __iter__(self):
        self.sampler.load_state_dict(self.sampler.state_at(self.start_update))
        for update in range(self.start_update, self.start_update + self.updates):
            yield from self.sampler.microbatches(update)
