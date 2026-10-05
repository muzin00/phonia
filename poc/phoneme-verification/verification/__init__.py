"""Phase 5 verification, reusing the Phase 4 frozen inference boundary."""

import sys
from pathlib import Path

REGISTRATION_ROOT = Path(__file__).resolve().parents[2] / "phoneme-user-registration"
for root in (REGISTRATION_ROOT, REGISTRATION_ROOT.parent / "phoneme-speaker-encoder"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from .inputs import IncompleteVerification, VerificationInput, load_verification_input
from .scoring import VerificationResult, load_result, save_result, verify

__all__ = [
    "IncompleteVerification",
    "VerificationInput",
    "VerificationResult",
    "load_result",
    "load_verification_input",
    "save_result",
    "verify",
]
