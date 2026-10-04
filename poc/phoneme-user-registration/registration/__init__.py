"""Phase 4 enrollment; Phase 5 should load profiles with the same FrozenEncoder."""

import sys
from pathlib import Path

# Reuse the frozen Phase 3 preprocessing/model implementation from the sibling PoC.
PHASE3_ROOT = Path(__file__).resolve().parents[2] / "phoneme-speaker-encoder"
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from .encoder import FrozenEncoder
from .profiles import (
    EnrollmentSegment,
    IncompleteEnrollment,
    UserProfile,
    load_profile,
    load_registration_input,
    register_user,
    save_profile,
)

__all__ = [
    "EnrollmentSegment",
    "FrozenEncoder",
    "IncompleteEnrollment",
    "UserProfile",
    "load_profile",
    "load_registration_input",
    "register_user",
    "save_profile",
]
