"""Deterministic Phase 3 input and selection primitives (design 2.0.0)."""

from .artifacts import compute_feature_statistics, make_enrollment, make_trials
from .input import InputPipeline, collate_segments, load_feature_statistics
from .manifest import Segment, load_segments, sha256_file
from .sampling import BalancedSampler, SampleRequest

__all__ = [
    "BalancedSampler",
    "InputPipeline",
    "SampleRequest",
    "Segment",
    "collate_segments",
    "compute_feature_statistics",
    "load_feature_statistics",
    "load_segments",
    "make_enrollment",
    "make_trials",
    "sha256_file",
]
